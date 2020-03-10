from pathlib import Path
from argparse import ArgumentParser

import Levenshtein
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TensorBoardLogger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from models.frontend import VisualFrontend
from models.backends import TransformerBackend
from utils import data


class Seq2SeqPretrainModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.vocab = data.CharVocab()
        self.model = TransformerBackend(self.vocab, nh=self.hparams.d_model)
        # self.frontend = VisualFrontend(512, resnet='resnet18')
        self.loss_fn = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=self.vocab.token2idx('<pad>'))

    def _increase_seq_len(self):
        a = self.train_ds.increase_seq_len()
        b = self.val_ds.increase_seq_len()

        assert a == b

        self.max_seq_len = a

    def on_epoch_start(self):
        # curriculum learning: increase maximum sequence length every
        # seq_inc_interval epochs
        # note that initial current_epoch is 0 so seq len is increased
        # from 1 to 2 before the training begins
        if self.hparams.seq_inc_interval > 0 and \
           self.current_epoch % self.hparams.seq_inc_interval == 0:
            self._increase_seq_len()
            print('max seq len is now', self.max_seq_len)

    def on_batch_end(self):
        self.logger.experiment.add_scalar('max_sequence_length',
                                          scalar_value=self.max_seq_len,
                                          global_step=self.global_step)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_nr):
        x, y, _ = batch

        pred = self(x, y)

        N = len(x)  # batch size

        pad = self.vocab.token2idx('<pad>')
        pad_column = torch.tensor([pad] * N).view(N, 1).type_as(y)

        # get target values from the training target by shifting by 1:
        target = torch.cat([y[:, 1:], pad_column], dim=1)

        loss = self.loss_fn(pred, target)

        log = {
            'loss/train': loss,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        if self.global_step % 200 == 0:
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

            header = '| Real    | Prediction    |  \n |--- | --- |  \n'

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

        return opt

    def validation_step(self, batch, batch_nr):
        x, y, _ = batch

        # x = self.frontend(x.unsqueeze(1))

        predictions = self.model(x)

        special = [
            self.vocab.token2idx(x) for x in ['<pad>', '<sos>', '<eos>']
        ]

        targets = [
            ''.join([
                self.vocab.idx2token(idx) for idx in label_seq
                if idx not in special
            ]) for label_seq in y
        ]

        cers = []

        for pred, target in zip(predictions, targets):
            cer = Levenshtein.distance(pred, target) / len(target)
            cers.append(cer)

        pred_string = '  \n'.join(
            [f'{label} | {pred}' for label, pred in zip(targets, predictions)])

        mean_cer = sum(cers) / len(cers)

        return {'val_loss': mean_cer, 'pred_string': pred_string}

    def validation_end(self, outputs):
        val_loss_mean = 0

        header = '| Real | Prediction | Greedy | \n | --- | --- | --- |  \n'

        s = header + '  \n'.join([output['pred_string'] for output in outputs])

        self.logger.experiment.add_text(tag='validation_predictions',
                                        text_string=s,
                                        global_step=self.current_epoch)

        for output in outputs:
            val_loss_mean += output['val_loss']

        val_loss_mean /= len(outputs)

        return {'val_loss': val_loss_mean, 'log': {'cer/valid': val_loss_mean}}

    @pl.data_loader
    def train_dataloader(self):
        # train_ds = data.PretrainFeatureDataset(self.hparams.data_root,
        #                                        vocab=self.vocab,
        #                                        easy=self.hparams.easy)
        train_ds = data.LRS2FeatureTrainSplit(
            Path(self.hparams.data_root, 'lrs2'), self.vocab)
        self.train_ds = train_ds
        self.max_seq_len = train_ds.max_seq_len

        train_dl = DataLoader(
            train_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda x: data.pad_collate(
                x, padding_value=self.vocab.token2idx('<pad>')),
            num_workers=self.hparams.workers,
            shuffle=True)

        return train_dl

    @pl.data_loader
    def val_dataloader(self):
        # val_ds = data.LRS2TestTrainDataset('train',
        #                                    os.path.join(
        #                                        self.hparams.data_root, 'lrs2'),
        #                                    vocab=self.vocab,
        #                                    transform=data.val_transform())
        val_ds = data.LRS2FeatureValSplit(Path(self.hparams.data_root, 'lrs2'),
                                          vocab=self.vocab)

        self.val_ds = val_ds

        val_dl = DataLoader(
            val_ds,
            batch_size=16,
            collate_fn=lambda x: data.pad_collate(
                x, padding_value=self.vocab.token2idx('<pad>')),
            num_workers=self.hparams.workers)

        return val_dl

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
        parser.add_argument('--seq_inc_interval', default=3, type=int)

        return parser


def main(hparams):
    module = Seq2SeqPretrainModule(hparams)

    save_dir = Path(__file__).parent.parent.absolute() / 'lightning_logs'
    experiment_name = 'train_test'
    version = int(hparams.checkpoint) if hparams.checkpoint else None

    logger = TensorBoardLogger(save_dir=save_dir,
                               name=experiment_name,
                               version=version)
    _ = logger.experiment  # create log dir

    base_path = save_dir / experiment_name / \
        f'version_{logger.version}'

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=True,
                                   mode='min')

    checkpoint_callback = ModelCheckpoint(filepath=base_path /
                                          'model_checkpoints',
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=True)

    trainer = pl.Trainer(
        logger=logger,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        log_gpu_memory='all',
        print_nan_grads=True,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        fast_dev_run=0,
        min_epochs=30,
        max_epochs=hparams.max_epochs,
        track_grad_norm=hparams.track_grad_norm,
        gradient_clip_val=1)

    # if hparams.frontend_weights:
    #     print('loading weights from a pretrained model...')
    #     module.frontend.load_state_dict(torch.load(hparams.frontend_weights))

    #     print('setting frontend requires_grad to False...')
    #     module.frontend.requires_grad = False

    trainer.fit(module)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--weight_hist', default=0, type=int)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--frontend_weights', default='', type=str)
    parser = Seq2SeqPretrainModule.add_model_specific_args(parser)

    hparams = parser.parse_args()
    main(hparams)
