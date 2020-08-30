import os
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from torch.utils.data import DataLoader

from models.models import PretrainClassifier
from utils import data
from utils.version import version as v


class VisualPretrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = PretrainClassifier()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)

        loss = self.criterion(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        log = {
            'loss/train': loss,
            'accuracy/train': accuracy,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            'epoch': self.current_epoch
        }

        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(),
                          lr=self.hparams.learning_rate,
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

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)

        loss = self.criterion(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([x['val_loss']
                                 for x in outputs]).sum() / len(outputs)
        val_acc = torch.tensor([x['val_acc']
                                for x in outputs]).sum() / len(outputs)

        return {
            'val_loss': 1 - val_acc,
            'log': {
                'loss/valid': val_loss,
                'accuracy/valid': val_acc
            }
        }

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)

        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'test_acc': accuracy}

    def test_epoch_end(self, outputs):
        test_acc = torch.tensor([x['test_acc']
                                 for x in outputs]).sum() / len(self.train_ds)

        return {'log': {'accuracy/test': test_acc}}

    def prepare_data(self):
        assert self.hparams.dataset in ['lrw1', 'pretrain']

        lrw1_root = os.path.join(self.hparams.data_root, 'lrw1')

        if self.hparams.dataset == 'lrw1':
            self.train_ds = data.LRW1Dataset(root=lrw1_root,
                                             splits='train',
                                             transform=data.train_transform())
            self.val_ds = data.LRW1Dataset(root=lrw1_root,
                                           splits='val',
                                           transform=data.val_transform())

        elif self.hparams.dataset == 'pretrain':
            lrs2_root = os.path.join(self.hparams.data_root, 'lrs2')

            lrw1_train = data.LRW1Dataset(root=lrw1_root,
                                          splits='train',
                                          transform=data.train_transform())
            lrs2_train = data.LRS2PretrainWordDataset(
                lrs2_root,
                classes=lrw1_train.classes,
                transform=data.train_transform())

            self.train_ds = lrw1_train + lrs2_train

            lrw1_val = data.LRW1Dataset(root=lrw1_root,
                                        splits='val',
                                        transform=data.val_transform())
            lrs2_val = data.LRS2PretrainWordDataset(
                lrs2_root,
                classes=lrw1_val.classes,
                transform=data.val_transform())

            self.val_ds = lrw1_val + lrs2_val

        self.test_ds = data.LRW1Dataset(root=lrw1_root,
                                        splits='test',
                                        transform=data.val_transform())

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          collate_fn=data.zero_pad,
                          num_workers=self.hparams.workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=2 * self.hparams.batch_size,
                          collate_fn=data.zero_pad,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=self.hparams.workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=2 * self.hparams.batch_size,
                          collate_fn=data.zero_pad,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=self.hparams.workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=64, type=int)

        # data
        parser.add_argument('--data_root', type=str, required=True)
        parser.add_argument('--dataset', type=str, default='lrw1')

        # training
        parser.add_argument('--workers', default=12, type=int)

        return parser


def main(hparams, version_hparams):
    print(hparams)

    module = VisualPretrainModule(hparams)

    logger = TensorBoardLogger(save_dir='tb_logs',
                               name='pretrain',
                               version=v(version_hparams, hparams))

    logger.log_hyperparams(hparams)

    ckpt_path = os.path.join(logger.save_dir, logger.name, logger.version,
                             '{epoch}-{val_loss:.2f}.ckpt')
    checkpoint_callback = ModelCheckpoint(ckpt_path,
                                          monitor='val_loss',
                                          verbose=True)

    trainer = pl.Trainer(logger=logger,
                         progress_bar_refresh_rate=100,
                         early_stop_callback=False,
                         checkpoint_callback=checkpoint_callback,
                         gpus=1,
                         log_gpu_memory='all',
                         min_epochs=hparams.epochs,
                         max_epochs=hparams.epochs,
                         track_grad_norm=hparams.track_grad_norm)

    trainer.fit(module)
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--weight_hist', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', type=str, default='')
    parser = VisualPretrainModule.add_model_specific_args(parser)

    hparams = parser.parse_args()

    # list of tuples (display_name, hparam_key, process_fn)
    version_hparams = [('lr', 'learning_rate'), ('bs', 'batch_size'),
                       ('ds', 'dataset'), ('description', None)]

    main(hparams, version_hparams)
