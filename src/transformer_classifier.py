import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
from torch.utils.data import DataLoader

from models.models import TransformerClassifier
from utils import data
from utils.version import version as v


class TransformerClassifierModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = TransformerClassifier(frontend=hparams.frontend,
                                           num_layers=hparams.num_layers,
                                           nhead=hparams.nhead,
                                           d_model=hparams.d_model)
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
                          lr=0,
                          weight_decay=self.hparams.weight_decay)

        sched = {
            'scheduler':
            optim.lr_scheduler.OneCycleLR(opt,
                                          max_lr=self.hparams.max_lr,
                                          epochs=self.hparams.max_epochs,
                                          steps_per_epoch=len(
                                              self.train_dataloader()),
                                          div_factor=self.hparams.div_factor),
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

        val_loss = torch.tensor([x['val_loss'].mean() for x in outputs]).mean()
        val_acc = torch.tensor([x['val_acc'].mean() for x in outputs]).mean()

        return {
            'val_acc': val_acc,
            'log': {
                'loss/valid': val_loss,
                'accuracy/valid': val_acc
            }
        }

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)

        loss = self.criterion(pred, y)
        accuracy = (pred.argmax(dim=1) == y).float().mean()

        return {'test_loss': loss, 'test_acc': accuracy}

    def test_epoch_end(self, outputs):

        test_loss = torch.tensor([x['test_loss'] for x in outputs]).mean()
        test_acc = torch.tensor([x['test_acc'] for x in outputs]).mean()

        return {'log': {'loss/test': test_loss, 'accuracy/test': test_acc}}

    def prepare_data(self):
        if hparams.resume_from_checkpoint:
            return

        root = os.path.join(self.hparams.data_root, 'lrw1')

        if self.hparams.train > 0:
            self.train_ds = data.LRW1Dataset(root=root,
                                             splits=['train'],
                                             transform=data.train_transform())
            self.val_ds = data.LRW1Dataset(root=root,
                                           splits='val',
                                           transform=data.val_transform())
            self.test_ds = data.LRW1Dataset(root=root,
                                            splits='test',
                                            transform=data.val_transform())

        else:
            self.train_ds = data.LRW1Dataset(root=root,
                                             splits=['train', 'val'],
                                             transform=data.train_transform())
            self.val_ds = data.LRW1Dataset(root=root,
                                           splits='test',
                                           transform=data.val_transform())
            self.test_ds = self.val_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          collate_fn=data.zero_pad,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=self.hparams.workers)

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
        parser.add_argument('--frontend', default='resnet', type=str)
        parser.add_argument('--num_layers', default=6, type=int)
        parser.add_argument('--nhead', default=8, type=int)
        parser.add_argument('--d_model', default=512, type=int)
        parser.add_argument('--div_factor', default=20, type=float)
        parser.add_argument('--max_lr', default=1e-4, type=float)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=64, type=int)

        # data
        parser.add_argument('--data_root', type=str)

        # training
        parser.add_argument('--workers', default=12, type=int)
        parser.add_argument('--train', default=1, type=int)

        return parser


def main(hparams, version_hparams):
    print('hparams', hparams)
    module = TransformerClassifierModule(hparams)

    logger = TensorBoardLogger(save_dir='new_logs',
                               name='transformer_classifier',
                               version=v(version_hparams, hparams))

    logger.log_hyperparams(hparams)

    ckpt_path = os.path.join(logger.save_dir, logger.name, logger.version,
                             '{epoch}-{val_acc:.2f}')
    checkpoint_callback = ModelCheckpoint(ckpt_path,
                                          save_top_k=1,
                                          monitor='val_acc',
                                          mode='max',
                                          verbose=True)

    trainer = pl.Trainer(logger=logger,
                         # profiler=AdvancedProfiler(),
                         # amp_level='O2',
                         # precision=16,
                         early_stop_callback=False,
                         checkpoint_callback=checkpoint_callback,
                         progress_bar_refresh_rate=100,
                         gpus=1,
                         log_gpu_memory='all',
                         min_epochs=hparams.min_epochs,
                         max_epochs=hparams.max_epochs,
                         resume_from_checkpoint=hparams.resume_from_checkpoint)

    trainer.fit(module)
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--description', type=str, default='')

    parser = TransformerClassifierModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    version_hparams = [('max_lr', None), ('bs', 'batch_size'),
                       ('description', None), ('max_epochs', None)]

    main(hparams, version_hparams)
