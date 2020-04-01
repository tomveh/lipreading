import os
from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.models import PretrainNet
from utils import data


class VisualPretrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = PretrainNet(resnet=hparams.resnet, nh=hparams.n_hidden)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nr):
        x, y = batch

        pred = self(x)

        loss = self.criterion(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        log = {
            'loss/train': loss,
            'accuracy/train': accuracy,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(),
                          lr=0,
                          weight_decay=self.hparams.weight_decay)

        sched = {
            'scheduler':
            optim.lr_scheduler.OneCycleLR(opt,
                                          max_lr=self.hparams.learning_rate,
                                          epochs=self.hparams.epochs,
                                          steps_per_epoch=len(
                                              self.train_dataloader()),
                                          div_factor=20),
            'interval':
            'step'
        }

        return [opt], [sched]

    def validation_step(self, batch, batch_nr):
        x, y = batch

        pred = self(x)

        loss = self.criterion(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        val_loss = 0
        val_acc = 0

        for output in outputs:
            val_loss += output['val_loss']
            val_acc += output['val_acc']

        val_loss /= len(outputs)
        val_acc /= len(outputs)

        return {
            'val_acc': val_acc,
            'log': {
                'loss/valid': val_loss,
                'accuracy/valid': val_acc
            }
        }

    def test_step(self, batch, batch_nr):
        x, y = batch

        pred = self(x)

        loss = self.criterion(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'test_loss': loss, 'test_acc': accuracy}

    def test_end(self, outputs):
        test_loss = 0
        test_acc = 0

        for output in outputs:
            test_loss += output['test_loss']
            test_acc += output['test_acc']

        test_loss /= len(outputs)
        test_acc /= len(outputs)

        return {'log': {'loss/test': test_loss, 'accuracy/test': test_acc}}

    def train_dataloader(self):
        if self.hparams.dataset == 'lrw1':
            train_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                        subdir='train',
                                        transform=data.train_transform())
        elif self.hparams.dataset == 'pretrain':
            lrw1 = data.LRW1Dataset(root=os.path.join(self.hparams.data_root,
                                                      'lrw1'),
                                    subdir='train',
                                    transform=data.val_transform())
            lrs2 = data.LRS2PretrainWordDataset(os.path.join(
                self.hparams.data_root, 'lrs2'),
                                                classes=lrw1.classes,
                                                transform=data.val_transform())

            train_ds = lrw1 + lrs2
        else:
            raise RuntimeError('unknown dataset')

        train_dl = DataLoader(train_ds,
                              batch_size=self.hparams.batch_size,
                              collate_fn=data.zero_pad,
                              num_workers=self.hparams.workers,
                              shuffle=True,
                              pin_memory=True)

        return train_dl

    def val_dataloader(self):
        if self.hparams.dataset == 'lrw1':
            root = self.hparams.data_root
        elif self.hparams.dataset == 'pretrain':
            root = os.path.join(self.hparams.data_root, 'lrw1')

        val_ds = data.LRW1Dataset(root=root,
                                  subdir='val',
                                  transform=data.val_transform())

        val_dl = DataLoader(val_ds,
                            batch_size=2 * self.hparams.batch_size,
                            collate_fn=data.zero_pad,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.hparams.workers)

        return val_dl

    def test_dataloader(self):
        if self.hparams.dataset == 'lrw1':
            root = self.hparams.data_root
        elif self.hparams.dataset == 'pretrain':
            root = os.path.join(self.hparams.data_root, 'lrw1')

        test_ds = data.LRW1Dataset(root=root,
                                   subdir='test',
                                   transform=data.val_transform())

        test_dl = DataLoader(test_ds,
                             batch_size=2 * self.hparams.batch_size,
                             collate_fn=data.zero_pad,
                             shuffle=False,
                             pin_memory=True,
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
        parser.add_argument('--dataset', type=str, default='lrw1')

        # training
        parser.add_argument('--workers', default=12, type=int)

        return parser


def main(hparams):
    print('hparams', hparams)
    module = VisualPretrainModule(hparams)

    save_dir = Path(__file__).parent.parent.absolute() / 'lightning_logs'
    experiment_name = 'pretrain'
    version = int(hparams.checkpoint) if hparams.checkpoint else None

    logger = TensorBoardLogger(save_dir=save_dir,
                               name=experiment_name,
                               version=version)
    _ = logger.experiment  # create log dir

    base_path = save_dir / experiment_name / \
        f'version_{logger.version}'

    early_stopping = EarlyStopping(monitor='val_acc',
                                   patience=3,
                                   verbose=True,
                                   mode='max')

    checkpoint_callback = ModelCheckpoint(filepath=base_path /
                                          'model_checkpoints',
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
                         min_epochs=hparams.epochs,
                         max_epochs=hparams.epochs,
                         track_grad_norm=hparams.track_grad_norm)

    trainer.fit(module)
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--fast_dev_run', default=0, type=int)
    parser.add_argument('--weight_hist', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', type=str, default='')
    parser = VisualPretrainModule.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)
