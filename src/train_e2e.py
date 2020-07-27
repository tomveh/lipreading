import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger
from torch.utils.data import random_split

from models.models import TransformerModel
from pretrain import VisualPretrainModule
from train import TrainModule
from utils import callbacks, data, data2
from utils.version import version as v


class EndToEndTrainModule(TrainModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = TransformerModel(self.vocab, 'resnet18')

    def prepare_data(self):
        ds = data2.LRS2PretrainDataset(
            os.path.join(self.hparams.data_root, 'lrs2'),
            self.vocab,
            transform=data.train_transform()) + data2.LRS3PretrainDataset(
                os.path.join(self.hparams.data_root, 'lrs3'),
                self.vocab,
                transform=data.train_transform())

        self.train_ds, self.val_ds = random_split(
            ds, [int(0.9 * len(ds)),
                 len(ds) - int(0.9 * len(ds))])

        for ds_i in self.train_ds.dataset.datasets:
            ds_i.max_seq_len = 3

    @staticmethod
    def add_model_specific_args(parser):
        parser = TrainModule.add_model_specific_args(parser)
        parser.add_argument('--frontend_weights', default='', type=str)
        parser.add_argument('--from_checkpoint', default='', type=str)
        return parser


def main(hparams, version_hparams):
    print(hparams)

    if hparams.from_checkpoint:
        module = EndToEndTrainModule.load_from_checkpoint(
            hparams.from_checkpoint)

        # loaded data_root is dependent on previous SLURM_JOB_ID so
        # replace it with a valid path
        checkpoint_version = [
            x for x in hparams.from_checkpoint.split('/')
            if x.startswith('version_')
        ][0].split('_')[1]

        p = Path(hparams.data_root)
        p.rename(p.parent / checkpoint_version)

        print('loaded module from checkpoint', hparams.from_checkpoint)
    else:
        module = EndToEndTrainModule(hparams)

    if hparams.frontend_weights:
        pretrained = VisualPretrainModule.load_from_checkpoint(
            hparams.frontend_weights)

        module.model.frontend.load_state_dict(
            pretrained.model.frontend.state_dict())

        print('loaded frontend weights from', hparams.frontend_weights)

    elif hparams.model_weights:
        m = EndToEndTrainModule.load_from_checkpoint(hparams.model_weights)

        module.model.load_state_dict(m.model.state_dict())

        print('loaded model weights from', hparams.model_weights)

    logger = TensorBoardLogger(save_dir='tb_logs',
                               name='train_e2e',
                               version=v(version_hparams, hparams))

    logger.log_hyperparams(hparams)

    ckpt_path = os.path.join(logger.save_dir, logger.name, logger.version,
                             '{epoch}-{val_loss:.2f}')
    checkpoint_callback = ModelCheckpoint(ckpt_path,
                                          save_top_k=1,
                                          monitor='val_loss',
                                          verbose=True)

    latest_ckpt = next(Path(hparams.resume_from_checkpoint).glob(
        '*.ckpt')) if hparams.resume_from_checkpoint else None

    trainer = pl.Trainer(
        logger=logger,
        amp_level='O2',
        precision=16,
        callbacks=[
            callbacks.PrintCallback(),
            callbacks.TimerCallback(),
            callbacks.PredictionLoggerCallback()
        ],
        progress_bar_refresh_rate=1000,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        gpus=int(hparams.gpus),
        log_gpu_memory='all',
        print_nan_grads=hparams.print_nan_grads,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        fast_dev_run=hparams.fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        resume_from_checkpoint=latest_ckpt,
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
