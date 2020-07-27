import os
from argparse import ArgumentParser

import Levenshtein
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from jiwer import wer
from models.backends import TransformerBackend
from utils import callbacks, data
from utils.version import version as v


class TrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.vocab = data.CharVocab()
        self.model = TransformerBackend(self.vocab)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=self.vocab.token2idx('<pad>'))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_nr):
        x, y, _ = batch

        pred = self(x, y[:, :-1])

        # drop <sos> token from each batch
        target = y[:, 1:]

        loss = self.loss_fn(pred, target)

        # save these for logging
        self.prev_loss = loss.item()
        self.target = target.detach()
        self.pred = pred.detach()

        log = {
            'loss/train': loss,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
            'train_ds_seq_len': self.train_ds.dataset.datasets[0].max_seq_len,
            'val_ds_seq_len': self.val_ds.dataset.datasets[0].max_seq_len,
            'epoch': self.current_epoch
        }

        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        return [opt]

    def validation_step(self, batch, batch_nr):
        # if self.current_epoch % self.hparams.valid_interval:
        #     return {}

        x, y, _ = batch

        beam_results = self(x)

        predictions = [
            max(result,
                key=lambda d: d['score'],
                default={'seq': '### FAILED TO DECODE ###'})['seq']
            for result in beam_results
        ]

        special = [
            self.vocab.token2idx(x) for x in ['<pad>', '<sos>', '<eos>']
        ]

        targets = [
            ''.join([
                self.vocab.idx2token(idx) for idx in targ if idx not in special
            ]) for targ in y
        ]

        cers = []
        wers = []

        for pred, target in zip(predictions, targets):
            cers.append(Levenshtein.distance(pred, target) / len(target))
            wers.append(wer(target, pred))

        pred_string = '  \n'.join([
            f'| {target} | {pred} | {wer} | '
            for target, pred, wer in zip(targets, predictions, wers)
        ])

        mean_cer = sum(cers) / len(cers)
        mean_wer = sum(wers) / len(wers)

        return {'cer': mean_cer, 'wer': mean_wer, 'pred_string': pred_string}

    def validation_epoch_end(self, outputs):
        # if self.current_epoch % self.hparams.valid_interval:
        #     self.prev_valid_stats = None
        #     return {'val_loss': float('inf')}

        header = '| Real | Prediction | WER | \n | --- | --- | --- | \n'

        s = header + '  \n'.join([output['pred_string'] for output in outputs])

        valid_cer = torch.tensor([x['cer'] for x in outputs]).mean()
        valid_wer = torch.tensor([x['wer'] for x in outputs]).mean()

        self.prev_valid_stats = {'cer': valid_cer, 'wer': valid_wer, 's': s}

        return {
            'val_loss': valid_wer,
            'log': {
                'valid/cer': valid_cer,
                'valid/wer': valid_wer
            }
        }

    def prepare_data(self):
        root = os.path.join(self.hparams.data_root, 'lrs2')
        ds = data.LRS2FeatureDataset(root, self.vocab)

        self.train_ds, self.val_ds = random_split(
            ds, [int(0.9 * len(ds)),
                 len(ds) - int(0.9 * len(ds))])

        # self.train_ds.dataset.max_seq_len = 3

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          collate_fn=lambda x: data.pad_collate(
                              x, padding_value=self.vocab.token2idx('<pad>')),
                          num_workers=self.hparams.workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams.batch_size,
                          collate_fn=lambda x: data.pad_collate(
                              x, padding_value=self.vocab.token2idx('<pad>')),
                          num_workers=self.hparams.workers,
                          shuffle=True,
                          pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--learning_rate', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=64, type=int)

        # data
        parser.add_argument('--data_root',
                            type=str,
                            default=f'/tmp/{os.environ["SLURM_JOB_ID"]}')

        # checkpoint
        parser.add_argument('--model_weights', default='', type=str)

        # training
        parser.add_argument('--workers', default=12, type=int)
        parser.add_argument('--valid_interval', default=10, type=int)
        parser.add_argument('--seq_inc_interval', default=500, type=int)

        # general
        parser.add_argument('--description', type=str, default='')

        return parser


def main(hparams, version_hparams):
    print(hparams)

    module = TrainModule(hparams)

    if hparams.model_weights:
        m = TrainModule.load_from_checkpoint(hparams.model_weights)
        module.model.load_state_dict(m.model.state_dict())
        print('loaded model weights from', hparams.model_weights)

    logger = TensorBoardLogger(save_dir='tb_logs',
                               name='train',
                               version=v(version_hparams, hparams))
    logger.log_hyperparams(hparams)

    ckpt_path = os.path.join(logger.save_dir, logger.name, logger.version,
                             '{epoch}-{val_loss:.2f}')
    checkpoint_callback = ModelCheckpoint(ckpt_path,
                                          monitor='val_loss',
                                          verbose=True)

    trainer = pl.Trainer(
        logger=logger,
        # amp_level='O2',
        # precision=16,
        callbacks=[
            callbacks.SequenceLengthCallback(),
            callbacks.PrintCallback(),
            callbacks.TimerCallback(),
            callbacks.PredictionLoggerCallback()
        ],
        progress_bar_refresh_rate=0,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        gpus=int(hparams.gpus),
        log_gpu_memory='all',
        print_nan_grads=hparams.print_nan_grads,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
    )

    trainer.fit(module)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser = TrainModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    version_hparams = [('lr', 'learning_rate'), ('bs', 'batch_size'),
                       ('gpus', None), ('description', None)]

    main(hparams, version_hparams)
