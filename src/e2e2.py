import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import Levenshtein
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from jiwer import wer
from models.models import TransformerModel
from utils import callbacks, data3
import tokenizers
from utils.version import version as v


class EndToEndTrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = tokenizers.Tokenizer.from_file(hparams.tokenizer)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=self.tokenizer.token_to_id('<pad>'))
        self.model = TransformerModel(self.vocab, 'resnet18')

        # set batch size as attribute so it can changed later
        self.batch_size = hparams.batch_size

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def prepare_data(self):
        ds = data3.LRS2PretrainDataset(
            os.path.join(self.hparams.data_root, 'lrs2'),
            self.vocab,
            transform=data3.train_transform()) + data3.LRS3PretrainDataset(
                os.path.join(self.hparams.data_root, 'lrs3'),
                self.vocab,
                transform=data3.train_transform())

        self.train_ds, self.val_ds = random_split(
            ds, [int(0.9 * len(ds)),
                 len(ds) - int(0.9 * len(ds))])

        for ds_i in self.train_ds.dataset.datasets:
            ds_i.max_seq_len = 4

    def training_step(self, batch, batch_nr):
        x, y, _ = batch

        pred = self(x, y[:, :-1])

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

        return opt

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

        targets = [self.vocab.decode(list(target)) for target in y]

        cers = []
        wers = []

        for pred, target in zip(predictions, targets):
            cers.append(Levenshtein.distance(pred, target) / len(target))
            wers.append(wer(target, pred))

        s = '  \n'.join([
            f'| {target} | {pred} | {wer} | {cer} |'
            for target, pred, wer, cer in zip(targets, predictions, wers, cers)
        ])

        return {'cers': cers, 'wers': wers, 's': s}

    def validation_epoch_end(self, outputs):
        # if self.current_epoch % self.hparams.valid_interval:
        #     self.prev_valid_stats = None
        #     return {'val_loss': float('inf')}

        header = '| Real | Pred | WER | CER |  \n | --- | --- | --- | --- |'
        s = header + '  \n'.join([output['s'] for output in outputs])

        cers = list(itertools.chain(*(x['cers'] for x in outputs)))
        wers = list(itertools.chain(*(x['wers'] for x in outputs)))

        valid_cer = torch.tensor(cers).mean()
        valid_wer = torch.tensor(wers).mean()

        self.prev_valid_stats = {'cer': valid_cer, 'wer': valid_wer, 's': s}

        return {
            'val_loss': valid_wer,
            'log': {
                'valid/cer': valid_cer,
                'valid/wer': valid_wer
            }
        }

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: data3.pad_collate(
                              x, tokenizer=self.tokenizer),
                          num_workers=self.hparams.workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          collate_fn=lambda x: data3.pad_collate(
                              x, tokenizer=self.tokenizer),
                          num_workers=self.hparams.workers,
                          shuffle=True,
                          pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--learning_rate', default=5e-5, type=float)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--tokenizer', type=str)

        # data
        parser.add_argument('--data_root', type=str)

        # continue training
        parser.add_argument('--load_from_checkpoint', type=str)
        parser.add_argument('--model_weights', default='', type=str)
        parser.add_argument('--frontend_weights', default='', type=str)

        # training
        parser.add_argument('--workers', default=12, type=int)
        # parser.add_argument('--valid_interval', default=10, type=int)
        # parser.add_argument('--seq_inc_interval', default=500, type=int)

        # general
        parser.add_argument('--description', type=str, default='')

        return parser


def main(hparams, version_hparams):
    print(hparams)

    if hparams.load_from_checkpoint:
        if hparams.load_from_checkpoint.endswith('.ckpt'):
            load_ckpt = hparams.load_from_checkpoint
        else:
            load_ckpt = next(Path(hparams.load_from_checkpoint).glob('*.ckpt'))

        module = EndToEndTrainModule.load_from_checkpoint(load_ckpt)
        module.batch_size = hparams.batch_size

        # load_from_checkpoint also loads data_root from hparams
        # so we have to replace the old slurm job id with current job id
        old_slurm_job_id = [
            x for x in hparams.load_from_checkpoint.split('/')
            if x.startswith('version_')
        ][0].split('_')[1]
        module.hparams.data_root = hparams.data_root.replace(
            old_slurm_job_id, os.environ["SLURM_JOB_ID"])

        print('loaded module from checkpoint', load_ckpt)
    elif hparams.model_weights:
        module = EndToEndTrainModule(hparams)
        load_ckpt = next(Path(hparams.model_weights).glob('*.ckpt'))
        m = EndToEndTrainModule.load_from_checkpoint(load_ckpt)
        module.model.load_state_dict(m.model.state_dict())

        print('loaded model weights from', hparams.model_weights)
    else:
        module = EndToEndTrainModule(hparams)

    if hparams.resume_from_checkpoint:
        resume_ckpt = next(Path(hparams.resume_from_checkpoint).glob('*.ckpt'))
        trainer = pl.Trainer(resume_from_checkpoint=resume_ckpt)
        trainer.fit(module)

    logger = TensorBoardLogger(save_dir='tb_logs',
                               name='e2e_subword',
                               version=v(version_hparams, hparams))

    logger.log_hyperparams(hparams)

    ckpt_path = os.path.join(logger.save_dir, logger.name, logger.version,
                             '{epoch}-{val_loss:.2f}')
    checkpoint_callback = ModelCheckpoint(ckpt_path,
                                          save_top_k=1,
                                          monitor='val_loss',
                                          verbose=True)

    resume_ckpt = next(Path(hparams.resume_from_checkpoint).glob(
        '*.ckpt')) if hparams.resume_from_checkpoint else None

    trainer = pl.Trainer(
        logger=logger,
        amp_level='O2',
        precision=16,
        callbacks=[
            callbacks.PrintCallback(),
            callbacks.TimerCallback(),
            callbacks.PredictionLoggerCallback(),
            callbacks.HackLRCallback(),
            # callbacks.CurriculumLearningCallback2(cls=EndToEndTrainModule)
        ],
        # overfit_pct=0.01,
        reload_dataloaders_every_epoch=True,
        progress_bar_refresh_rate=0,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        gpus=int(hparams.gpus),
        log_gpu_memory='all',
        print_nan_grads=hparams.print_nan_grads,
        accumulate_grad_batches=1,
        fast_dev_run=hparams.fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        track_grad_norm=hparams.track_grad_norm)

    trainer.fit(module)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser = EndToEndTrainModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    hparams = parser.parse_args()

    version_hparams = [('lr', 'learning_rate'), ('bs', 'batch_size'),
                       ('gpus', None), ('description', None)]

    main(hparams, version_hparams)
