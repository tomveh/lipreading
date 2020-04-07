import os
from argparse import ArgumentParser

import Levenshtein
import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models.backends import TransformerBackend
from utils import data


class Seq2SeqPretrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.vocab = data.CharVocab()
        self.model = TransformerBackend(self.vocab, nh=self.hparams.d_model)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=self.vocab.token2idx('<pad>'))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_nr):
        x, y, _ = batch

        # drop the last column which contains pad or eos token because
        # we don't care about the next token after eos (if the dropped
        # token is pad and eos is still in the train sequence, then
        # the corresponding target token will be pad and the loss will
        # ignore the prediction matching the eos token)
        y_ = y[:, :-1]

        pred = self(x, y_)

        # drop <sos> token from each batch
        target = y[:, 1:]

        loss = self.loss_fn(pred, target)

        log = {
            'loss/train': loss,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        if self.global_step % 5000 == 0:
            target_s = [
                ''.join([self.vocab.idx2token(idx) for idx in line
                         ]).replace('<sos>',
                                    '$').replace('<eos>',
                                                 '$').replace('<pad>', '#')
                for line in target
            ]

            pred_s = [
                ''.join([self.vocab.idx2token(idx)
                         for idx in line]).replace('<eos>', '$')
                for line in pred.argmax(dim=1)
            ]

            header = '| Label | Pred |  \n | --- | --- |  \n'

            s = header + '  \n'.join([
                f'{label} | {pred} |' for label, pred in zip(target_s, pred_s)
            ])

            self.logger.experiment.add_text('train_prediction',
                                            s,
                                            global_step=self.global_step)

        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        sched = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                     factor=0.5,
                                                     patience=50)

        return [opt], [sched]

    def validation_step(self, batch, batch_nr):
        x, y, _ = batch

        predictions = self(x)

        greedys = self.model.greedy(x)

        special = [
            self.vocab.token2idx(x) for x in ['<pad>', '<sos>', '<eos>']
        ]

        targets = [
            ''.join([
                self.vocab.idx2token(idx) for idx in targ if idx not in special
            ]) for targ in y
        ]

        cers = []

        for pred, target in zip(predictions, targets):
            cer = Levenshtein.distance(pred, target) / len(target)
            cers.append(cer)

        pred_string = '  \n'.join([
            f'{label} | {pred} | {greedy}'
            for label, pred, greedy in zip(targets, predictions, greedys)
        ])

        mean_cer = sum(cers) / len(cers)

        return {'val_loss': mean_cer, 'pred_string': pred_string}

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0

        header = '| Real | Prediction | Greedy |  \n | --- | --- | --- |  \n'

        s = header + '  \n'.join(
            [output['pred_string'] for output in outputs[:10]])

        self.logger.experiment.add_text(tag='validation_predictions',
                                        text_string=s,
                                        global_step=self.current_epoch)

        for output in outputs:
            val_loss_mean += output['val_loss']

        val_loss_mean /= len(outputs)

        return {'val_loss': val_loss_mean, 'log': {'cer/valid': val_loss_mean}}

    def prepare_data(self):
        root = os.path.join(self.hparams.data_root, 'lrs2')
        ds = data.LRS2FeatureDataset(root, self.vocab)

        self.train_ds, self.val_ds = random_split(
            ds, [int(0.8 * len(ds)),
                 len(ds) - int(0.8 * len(ds))])

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          collate_fn=lambda x: data.pad_collate(
                              x, padding_value=self.vocab.token2idx('<pad>')),
                          num_workers=self.hparams.workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams.batch_size,
                          collate_fn=lambda x: data.pad_collate(
                              x, padding_value=self.vocab.token2idx('<pad>')),
                          num_workers=self.hparams.workers,
                          shuffle=True)

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

        # training
        parser.add_argument('--workers', default=16, type=int)
        # parser.add_argument('--seq_inc_interval', default=3, type=int)

        return parser


def main(hparams):
    module = Seq2SeqPretrainModule(hparams)

    logger = TestTubeLogger(save_dir='tt_logs',
                            name='train',
                            description=hparams.description,
                            debug=False)

    trainer = pl.Trainer(
        logger=logger,
        early_stop_callback=False,
        checkpoint_callback=True,
        gpus=1,
        log_gpu_memory='all',
        print_nan_grads=True,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        fast_dev_run=0,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        track_grad_norm=hparams.track_grad_norm)

    trainer.fit(module)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--min_epochs', default=1, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--description', type=str, default='')
    parser = Seq2SeqPretrainModule.add_model_specific_args(parser)

    hparams = parser.parse_args()
    main(hparams)
