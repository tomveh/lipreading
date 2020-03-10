from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.models import PretrainNet
from utils import data


class VisualPretrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = PretrainNet(resnet=hparams.resnet, nh=hparams.n_hidden)

    def forward(self, x):
        return self.model((x))

    def training_step(self, batch, batch_nr):
        x, y = batch
        pred = self.forward(x)

        loss = F.cross_entropy(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        log = {
            'loss/train': loss,
            'accuracy/train': accuracy,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        return {'loss': loss, 'log': log}

    def on_after_backward(self):
        if (self.hparams.weight_hist > 0) and (self.global_step % 100 == 0):
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_histogram(
                        f'grad-{name}',
                        param.grad,
                        global_selftep=self.global_step)
                    self.logger.experiment.add_histogram(
                        f'weight-{name}',
                        param.data,
                        global_step=self.global_step)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        return opt

    def validation_step(self, batch, batch_nr):
        x, y = batch

        pred = self.forward(x)

        loss = F.cross_entropy(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0

        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        return {
            'val_acc': val_acc_mean,
            'log': {
                'loss/valid': val_loss_mean,
                'accuracy/valid': val_acc_mean
            }
        }

    def test_step(self, batch, batch_nr):
        x, y = batch

        pred = self.forward(x)

        loss = F.cross_entropy(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'test_loss': loss, 'test_acc': accuracy}

    def test_end(self, outputs):
        test_loss_mean = 0
        test_acc_mean = 0

        for output in outputs:
            test_loss_mean += output['test_loss']
            test_acc_mean += output['test_acc']

        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)

        return {
            'log': {
                'loss/test': test_loss_mean,
                'accuracy/test': test_acc_mean
            }
        }

    @pl.data_loader
    def train_dataloader(self):
        train_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                    subdir='train',
                                    transform=data.train_transform())

        train_dl = DataLoader(train_ds,
                              batch_size=self.hparams.batch_size,
                              shuffle=True,
                              num_workers=self.hparams.workers)

        return train_dl

    @pl.data_loader
    def val_dataloader(self):
        val_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                  subdir='val',
                                  transform=data.val_transform())

        val_dl = DataLoader(val_ds,
                            batch_size=2 * self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.workers)

        return val_dl

    @pl.data_loader
    def test_dataloader(self):

        test_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                   subdir='test',
                                   transform=data.val_transform())

        test_dl = DataLoader(test_ds,
                             batch_size=2 * self.hparams.batch_size,
                             shuffle=False,
                             num_workers=self.hparams.workers)

        return test_dl

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--resnet', type=str)
        parser.add_argument('--n_hidden', type=int, default=512)
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=64, type=int)

        # data
        parser.add_argument('--data_root', type=str, required=True)

        # training
        parser.add_argument('--workers', default=16, type=int)

        return parser


def main(hparams):
    module = VisualPretrainModule(hparams)

    save_dir = Path(__file__).parent.parent.absolute() / 'lightning_logs'
    experiment_name = 'pretrain'
    version = int(hparams.checkpoint) if hparams.checkpoint else None

    logger = TestTubeLogger(save_dir=save_dir,
                            name=experiment_name,
                            version=version,
                            description=hparams.description)

    base_path = save_dir / experiment_name / \
        f'version_{logger.experiment.version}'

    early_stopping = EarlyStopping(monitor='val_acc',
                                   patience=3,
                                   verbose=True,
                                   mode='max')

    checkpoint_callback = ModelCheckpoint(filepath=base_path / 'checkpoints',
                                          monitor='val_acc',
                                          mode='max',
                                          verbose=True)

    trainer = pl.Trainer(logger=logger,
                         early_stop_callback=early_stopping,
                         checkpoint_callback=checkpoint_callback,
                         show_progress_bar=True,
                         gpus=1,
                         log_gpu_memory='all',
                         fast_dev_run=hparams.fast_dev_run,
                         max_nb_epochs=hparams.max_epochs,
                         track_grad_norm=hparams.track_grad_norm)

    trainer.fit(module)
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--fast_dev_run', default=0, type=int)
    parser.add_argument('--weight_hist', default=0, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--checkpoint', type=str)
    parser = VisualPretrainModule.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)
