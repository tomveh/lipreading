import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.models import LipreadingResNet
from util.data import LRW1Dataset
from util.transforms import Map, Normalize, RandomFrameDrop
import util.videotransforms.video_transforms as video_transforms


class LipreadingClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = LipreadingResNet(hparams.resnet, hparams.backend)
        self.__init_weights()

    def __init_weights(self):
        for name, module in self.model.named_modules():
            if 'Conv2d' in module.__class__.__name__:
                nn.init.kaiming_uniform_(module.weight.data)

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
            'val_loss': val_loss_mean,
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

    def train_transform(self):
        return transforms.Compose([
            lambda t: t.permute(0, 3, 1, 2),
            RandomFrameDrop(0.05), torch.unbind,
            Map(
                transforms.Compose(
                    [transforms.ToPILImage(),
                     transforms.Grayscale()])),
            video_transforms.CenterCrop([122, 122]),
            video_transforms.RandomCrop([112, 112]),
            video_transforms.RandomHorizontalFlip(),
            Map(transforms.ToTensor()), torch.stack,
            lambda t: t.transpose(0, 1),
            Normalize()
        ])

    def val_transform(self):
        return transforms.Compose([
            lambda t: t.permute(0, 3, 1, 2), torch.unbind,
            Map(
                transforms.Compose(
                    [transforms.ToPILImage(),
                     transforms.Grayscale()])),
            video_transforms.CenterCrop([112, 112]),
            Map(transforms.ToTensor()), torch.stack,
            lambda t: t.transpose(0, 1),
            Normalize()
        ])

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.augment >= 1:
            transform = self.train_transform()
        elif self.hparams.augment < 1:
            transform = self.val_transform()
        else:
            raise RuntimeError(
                f'invalid hparams.augment value {self.hparams.augment}')

        train_ds = LRW1Dataset(root=self.hparams.data_root,
                               subdir='train',
                               loader=lambda filename: torchvision.io.
                               read_video(filename, pts_unit='sec')[0],
                               transform=transform)

        train_dl = DataLoader(train_ds,
                              batch_size=self.hparams.batch_size,
                              shuffle=True,
                              num_workers=self.hparams.workers)

        return train_dl

    @pl.data_loader
    def val_dataloader(self):
        transform = self.val_transform()

        val_ds = LRW1Dataset(root=self.hparams.data_root,
                             subdir='val',
                             loader=lambda filename: torchvision.io.read_video(
                                 filename, pts_unit='sec')[0],
                             transform=transform)

        val_dl = DataLoader(val_ds,
                            batch_size=2 * self.hparams.batch_size,
                            shuffle=True,
                            num_workers=self.hparams.workers)

        return val_dl

    @pl.data_loader
    def test_dataloader(self):
        transform = self.val_transform()

        test_ds = LRW1Dataset(root=self.hparams.data_root,
                              subdir='test',
                              loader=lambda filename: torchvision.io.
                              read_video(filename, pts_unit='sec')[0],
                              transform=transform)

        test_dl = DataLoader(test_ds,
                             batch_size=2 * self.hparams.batch_size,
                             shuffle=True,
                             num_workers=self.hparams.workers)

        return test_dl

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--resnet', type=str)
        parser.add_argument('--backend', type=str)
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=64, type=int)

        # data
        parser.add_argument('--data_root', type=str, required=True)
        parser.add_argument('--augment', type=int, default=1)

        # training
        parser.add_argument('--workers', default=16, type=int)
        parser.add_argument('--weight_hist', default=0, type=int)
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--checkpoint', type=str)
        parser.add_argument('--fast_dev_run', default=0, type=int)
        parser.add_argument('--frontend_weights', type=str)

        return parser


def main(hparams):
    module = LipreadingClassifier(hparams)

    if hparams.checkpoint:
        print(f'loading from checkpoint {hparams.checkpoint}...')
        logger = TestTubeLogger(save_dir='lightning_logs',
                                version=hparams.checkpoint)
        trainer = pl.Trainer(logger=logger)
    else:
        trainer = pl.Trainer(train_percent_check=0.001,
                             val_percent_check=0.1,
                             show_progress_bar=True,
                             gpus=1,
                             log_gpu_memory='all',
                             fast_dev_run=hparams.fast_dev_run,
                             max_nb_epochs=hparams.max_epochs,
                             track_grad_norm=hparams.track_grad_norm)

    if hparams.frontend_weights:
        print('loading weights from a pretrained model...')
        module.model.frontend.load_state_dict(
            torch.load(hparams.frontend_weights))

        print('setting frontend requires_grad to False...')
        for param in module.model.frontend.parameters():
            param.requires_grad = False

    trainer.fit(module)
    trainer.test()

    path = os.path.join(trainer.logger.save_dir, 'lightning_logs',
                        f'version_{trainer.logger.version}',
                        'frontend-weights.pt')

    torch.save(module.model.frontend.state_dict(), path)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser = LipreadingClassifier.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)
