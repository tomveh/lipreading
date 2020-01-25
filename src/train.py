from pathlib import Path
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util import data
from models.frontend import VisualFrontEnd
from models.backends import TransformerBackend


class TransformerModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        frontend = VisualFrontEnd(out_channels=hparams.d_model,
                                  resnet='resnet18')

        n_vocab = 40  # TODO
        self.model = TransformerBackend(n_vocab=n_vocab,
                                        d_model=hparams.d_model,
                                        frontend=frontend)
        self.loss_fn = nn.CrossEntropyLoss()
        self.text_summary = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_nr):
        x, y = batch

        pred = self.forward(x, y)

        batch_size = len(x)

        # get target values from the training target by shifting by 1:
        # drop <sos> and add <pad> to the end of every sequence
        target = torch.cat([
            y[:, 1:],
            torch.tensor([self.vocab['<pad>']] * batch_size).view(
                batch_size, 1).type_as(y)
        ],
                           dim=1)

        loss = self.loss_fn(pred, target)

        log = {
            'loss/train': loss,
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

        pred = self.model.inference(x, self.vocab)

        batch_size = len(x)
        max_seq_len = max(y.shape[1], pred.shape[1])

        # make two tensors of the same shape to compare prediction and label
        pred_long = torch.ones([batch_size, max_seq_len]) * self.vocab['<pad>']
        pred_long[:, :pred.shape[1]] = pred
        y_long = torch.ones([batch_size, max_seq_len]) * self.vocab['<pad>']
        y_long[:, :y.shape[1]] = y

        accuracy = (y_long == pred_long).all(dim=1).float().mean()

        if self.text_summary:

            def lookup(idx):
                return self.inverse_vocab[idx.item()]

            preds = [''.join(lookup(idx) for idx in line) for line in pred]
            labels = [''.join(lookup(idx) for idx in line) for line in y]

            header = '| Real    | Prediction    |  \n |--- | --- |  \n'
            string = header + '  \n'.join(
                [f'{label} | {pred} |' for label, pred in zip(labels, preds)])
            string = string.replace('<sos>', 's').replace('<pad>', 'p')

            self.logger.experiment.add_text(tag='prediction',
                                            text_string=string,
                                            global_step=self.global_step)

            self.text_summary = False

        return {'val_acc': accuracy}

    def validation_end(self, outputs):
        val_acc_mean = 0

        for output in outputs:
            val_acc_mean += output['val_acc']

        val_acc_mean /= len(outputs)

        self.text_summary = True

        return {
            'val_acc': val_acc_mean,
            'log': {
                'accuracy/valid': val_acc_mean
            }
        }

    def test_step(self, batch, batch_nr):
        x, y = batch

        pred = self.model.inference(x, vocab=self.vocab)

        batch_size = len(x)
        max_seq_len = max(y.shape[1], pred.shape[1])

        pred_long = torch.ones([batch_size, max_seq_len]) * self.vocab['<pad>']
        pred_long[:, :pred.shape[1]] = pred
        y_long = torch.ones([batch_size, max_seq_len]) * self.vocab['<pad>']
        y_long[:, :y.shape[1]] = y

        accuracy = (y_long == pred_long).all(dim=1).float().mean()

        return {'test_acc': accuracy}

    def test_end(self, outputs):
        test_acc_mean = 0

        for output in outputs:
            test_acc_mean += output['test_acc']

        test_acc_mean /= len(outputs)

        return {'log': {'accuracy/test': test_acc_mean}}

    @pl.data_loader
    def train_dataloader(self):
        train_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                    subdir='train',
                                    loader=lambda path: torchvision.io.
                                    read_video(path, pts_unit='sec')[0],
                                    transform=data.train_transform(),
                                    easy=self.hparams.easy,
                                    classification=False)

        self.vocab = dict(
            (token, i) for i, token in enumerate(train_ds.vocab.tokens))
        self.inverse_vocab = dict(
            (i, token) for token, i in self.vocab.items())

        train_dl = DataLoader(train_ds,
                              batch_size=self.hparams.batch_size,
                              shuffle=True,
                              collate_fn=lambda x: data.pad_collate(
                                  x,
                                  padding_value=self.vocab['<pad>'],
                                  sos_value=self.vocab['<sos>']),
                              num_workers=self.hparams.workers)

        return train_dl

    @pl.data_loader
    def val_dataloader(self):
        val_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                  subdir='val',
                                  loader=lambda path: torchvision.io.
                                  read_video(path, pts_unit='sec')[0],
                                  transform=data.val_transform(),
                                  easy=self.hparams.easy,
                                  classification=False)

        val_dl = DataLoader(val_ds,
                            batch_size=2 * self.hparams.batch_size,
                            shuffle=True,
                            collate_fn=lambda x: data.pad_collate(
                                x,
                                padding_value=self.vocab['<pad>'],
                                sos_value=self.vocab['<sos>']),
                            num_workers=self.hparams.workers)

        return val_dl

    @pl.data_loader
    def test_dataloader(self):
        test_ds = data.LRW1Dataset(root=self.hparams.data_root,
                                   subdir='test',
                                   loader=lambda path: torchvision.io.
                                   read_video(path, pts_unit='sec')[0],
                                   transform=data.val_transform(),
                                   easy=self.hparams.easy,
                                   classification=False)

        test_ds = DataLoader(test_ds,
                             batch_size=2 * self.hparams.batch_size,
                             shuffle=True,
                             collate_fn=lambda x: data.pad_collate(
                                 x,
                                 padding_value=self.vocab['<pad>'],
                                 sos_value=self.vocab['<sos>']),
                             num_workers=self.hparams.workers)

        return test_ds

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--d_model', default=512, type=int)
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=64, type=int)

        # data
        parser.add_argument('--data_root', type=str, required=True)
        parser.add_argument('--easy', default=0, type=int)

        # training
        parser.add_argument('--workers', default=16, type=int)
        parser.add_argument('--weight_hist', default=0, type=int)
        parser.add_argument('--checkpoint', default='', type=str)
        parser.add_argument('--fast_dev_run', default=0, type=int)
        parser.add_argument('--frontend_weights', default='', type=str)

        return parser


def main(hparams):
    module = TransformerModel(hparams)

    save_dir = Path('.') / 'lightning_logs'
    experiment_name = 'train'
    version = int(hparams.checkpoint) if hparams.checkpoint else None

    logger = TestTubeLogger(save_dir=save_dir,
                            name=experiment_name,
                            debug=hparams.fast_dev_run > 0,
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

    trainer = pl.Trainer(
        logger=logger,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        log_gpu_memory='all',
        print_nan_grads=True,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        fast_dev_run=hparams.fast_dev_run,
        max_nb_epochs=hparams.max_epochs,
        track_grad_norm=hparams.track_grad_norm)

    if hparams.frontend_weights:
        print('loading weights from a pretrained model...')
        module.model.frontend.load_state_dict(
            torch.load(hparams.frontend_weights))

        print('setting frontend requires_grad to False...')
        module.model.frontend.requires_grad = False

    trainer.fit(module)
    trainer.test()


#     path = os.path.join(trainer.logger.save_dir, 'lightning_logs',
#                         f'version_{trainer.logger.version}',
#                         'frontend-weights.pt')
#     torch.save(module.model.frontend.state_dict(), path)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--description', type=str, default='')
    parser = TransformerModel.add_model_specific_args(parser)

    hparams = parser.parse_args()
    main(hparams)
