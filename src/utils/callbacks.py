import time
from pathlib import Path

import pytorch_lightning as pl


class TimerCallback(pl.Callback):
    def on_batch_start(self, trainer, pl_module):
        self.batch_start = time.time()

    def on_batch_end(self, trainer, pl_module):
        end = time.time()

        t = end - self.batch_start

        if pl_module.global_step % (trainer.row_log_interval * 10) == 0:
            pl_module.logger.experiment.add_scalar(
                'time/time_per_batch', t, global_step=pl_module.global_step)

    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_epoch_end(self, trainer, pl_module):
        end = time.time()

        t = end - self.epoch_start

        pl_module.logger.experiment.add_scalar(
            'time/epoch_time_min', t / 60, global_step=pl_module.global_step)


class PrintCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print('Started training...')

    def on_epoch_end(self, trainer, pl_module):
        stats = {}

        stats['epoch'] = pl_module.current_epoch
        stats['loss'] = pl_module.prev_loss

        if pl_module.prev_valid_stats is not None:
            stats['wer'] = pl_module.prev_valid_stats['wer']
            stats['cer'] = pl_module.prev_valid_stats['cer']

        s = ' | '.join([
            f'{key}: {value:{".4f" if type(value) == float else ""}}'
            for key, value in stats.items()
        ])

        print(s)


class PredictionLoggerCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # # log train predictions every 10 epochs
        # if pl_module.current_epoch % 10 == 0:

        #     targets = [
        #         pl_module.vocab.decode(list(y)) for y in pl_module.target
        #     ]

        #     preds = []

        #     for line in pl_module.pred.argmax(dim=1):
        #         predicted_string = ''.join([
        #             pl_module.vocab.idx2token(idx) for idx in line
        #         ]).replace('<eos>', '$')

        #         pred = pl_module.vocab.decode()

        #         print('pred', predicted_string)

        #         predicted_string = pl_module.vocab.decode(
        #             [idx for idx in line]).replace('<eos>', '$')

        #         preds.append(predicted_string)

        #     header = '| Target | Predicted |  \n | --- | --- |  \n'

        #     s = header + '  \n'.join([
        #         f'| {target} | {pred} |'
        #         for target, pred in zip(targets, preds)
        #     ])

        #     pl_module.logger.experiment.add_text(
        #         'train_prediction', s, global_step=pl_module.current_epoch)

        # log validation predictions
        if pl_module.prev_valid_stats is not None:
            pl_module.logger.experiment.add_text(
                tag='validation_predictions',
                text_string=pl_module.prev_valid_stats['s'],
                global_step=pl_module.current_epoch)


class CurriculumLearningCallback(pl.Callback):
    def __init__(self):
        super().__init__()

        with open('pretrain_schedule.csv', 'r') as f:
            lines = f.read().splitlines()

            header = lines[0].split(',')
            rest = (line.split(',') for line in lines[1:])

            # convert strings to correct type
            #
            # the csv headers are max_seq_len, batch_size,
            # max_clip_length, accumulate so the corresponding types
            # are int, int, float, int
            rest_ = ([f(x) for f, x in zip([int, int, float, int], line)]
                     for line in rest)

            self.pretrain_schedule = {
                line[0]: dict(zip(header[1:], line[1:]))
                for line in rest_
            }

    def on_train_start(self, trainer, pl_module):
        # set gradient accumulation
        acc_schedule = {
            key: value['accumulate']
            for key, value in self.pretrain_schedule.items()
        }

        trainer.configure_accumulated_gradients(acc_schedule)

        # set max clip lengths for lrs2 and lrs3 (assuming that the ds
        # is a ConcatDataset)
        for ds in pl_module.train_ds.dataset.datasets:
            ds.max_lengths = {
                key: value['max_clip_length']
                for key, value in self.pretrain_schedule.items()
            }

    def on_epoch_start(self, trainer, pl_module):
        print('this is epoch', pl_module.current_epoch)

    def on_epoch_end(self, trainer, pl_module):
        max_seq_len = pl_module.train_ds.dataset.datasets[0].max_seq_len
        acc = self.pretrain_schedule[max_seq_len]['accumulate']

        # if not divisible then don't change anything
        if pl_module.current_epoch % (acc *
                                      pl_module.hparams.seq_inc_interval):
            return

        # first increase the max sequence length
        for ds in pl_module.train_ds.dataset.datasets:
            ds.max_seq_len += 1
            max_len = ds.max_seq_len

        sched = self.pretrain_schedule[max_len]

        if sched:
            # set batch size for next epoch (dl is reloaded in the
            # beginning of each epoch)
            pl_module.batch_size = sched['batch_size']


class HackLRCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.lr = pl_module.hparams.learning_rate

    def on_batch_start(self, trainer, pl_module):
        lr_epochs = 10
        div_factor = 20
        epochs_per_batch = len(pl_module.train_dataloader())

        init_lr = self.lr / div_factor

        lr_multiplier = pl_module.global_step / (epochs_per_batch * lr_epochs)

        trainer.optimizers[0].param_groups[0]['lr'] = min(
            init_lr + lr_multiplier * self.lr, self.lr)


class CurriculumLearningCallback2(pl.Callback):
    def __init__(self, cls):
        self.cls = cls
        super().__init__()

        with open('pretrain_schedule.csv', 'r') as f:
            lines = f.read().splitlines()

            header = lines[0].split(',')
            rest = (line.split(',') for line in lines[1:])

            # convert strings to correct type
            #
            # the csv headers are max_seq_len, batch_size,
            # max_clip_length, accumulate so the corresponding types
            # are int, int, float, int
            rest_ = ([f(x) for f, x in zip([int, int, float, int], line)]
                     for line in rest)

            self.pretrain_schedule = {
                line[0]: dict(zip(header[1:], line[1:]))
                for line in rest_
            }

        self.losses = []

    def on_train_start(self, trainer, pl_module):
        # set max clip lengths for lrs2 and lrs3 (assuming that the ds
        # is a ConcatDataset)
        for ds in pl_module.train_ds.dataset.datasets:
            ds.max_lengths = {
                key: value['max_clip_length']
                for key, value in self.pretrain_schedule.items()
            }

    def on_epoch_start(self, trainer, pl_module):
        print('this is epoch', pl_module.current_epoch)

    def on_epoch_end(self, trainer, pl_module):
        # the csv headers are max_seq_len, batch_size,
        # max_clip_length, accumulate so the corresponding types
        # are int, int, float, int

        print('epoch', pl_module.current_epoch, 'end')

        if len(self.losses) < 10:
            print('less than 10')
            self.losses.append(pl_module.prev_loss)
            return

        # if loss is smaller than previous losses then continue training
        if pl_module.prev_loss < min(self.losses[-10:]):
            print('more than 10 but smaller loss')
            return

        print('next seq len')

        # if valid loss has not decreased for 10 epochs then increase seq len

        # there should be just 1 ckpt file which has the best weights
        # TODO: but it might have valid loss computed on shorter seq len
        ckpt_path = next(
            Path('/scratch/work/vehvilt2/lipreading',
                 pl_module.logger.save_dir, pl_module.logger.name,
                 pl_module.logger.version).glob('*.ckpt'))

        old_weights = self.cls.load_from_checkpoint(ckpt_path).state_dict()
        pl_module.load_state_dict(old_weights)

        print('loaded weights from', ckpt_path)

        # first increase the max sequence length
        for ds in pl_module.train_ds.dataset.datasets:
            ds.max_seq_len += 1
            max_seq_len = ds.max_seq_len

        print('max seq len increased to', max_seq_len)

        acc = self.pretrain_schedule[max_seq_len]['accumulate']
        trainer.configure_accumulated_gradients(acc)

        print('accumulate', acc)

        pl_module.batch_size = self.pretrain_schedule[max_seq_len][
            'batch_size']

        print('batch size is', pl_module.batch_size)

        # reset losses
        self.losses = []
