import math
import os
from collections import defaultdict
from pathlib import Path
from random import randrange

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

import utils.transforms as t


def video_loader(path, start_pts=0, end_pts=None):
    video, audio, info = torchvision.io.read_video(str(path),
                                                   start_pts=start_pts,
                                                   end_pts=end_pts,
                                                   pts_unit='sec')

    return video


def train_transform():
    return transforms.Compose([
        t.Permute(),
        t.Crop((112, 112), 'random'),
        t.RandomHorizontalFlip(),
        t.GrayScale(),
        t.Normalize(),
        t.RandomFrameDrop(0.05),
        t.Transpose()
    ])


def val_transform():
    return transforms.Compose([
        t.Permute(),
        t.Crop((112, 112), 'center'),
        t.GrayScale(),
        t.Normalize(),
        t.Transpose()
    ])


def zero_pad(samples):
    x, y = zip(*samples)

    padded = torch.nn.utils.rnn.pad_sequence(x,
                                             batch_first=True,
                                             padding_value=0)

    return padded.unsqueeze(1), torch.stack(y)


def pad_collate(samples, padding_value):
    """Add padding to samples to make equal length

    :param samples: tuple (x, y, *rest) where the video tensor x is
    padded with zeros and the transcript tensor y is padded with
    `padding_value`
    :param padding_value: value used to pad y
    :returns: tuple (padded_x, padded_y, rest)

    """
    x, y, *rest = zip(*samples)

    padded_x = torch.nn.utils.rnn.pad_sequence(x,
                                               batch_first=True,
                                               padding_value=0)
    padded_y = torch.nn.utils.rnn.pad_sequence(y,
                                               batch_first=True,
                                               padding_value=padding_value)
    return padded_x, padded_y, rest


class CharVocab:
    def __init__(self):

        # Vocab has 26 chars, 10 digits, space, apostophe, <eos>, <sos>, <pad>
        special = ['<eos>', ' ', "'"]
        chars = [chr(i) for i in range(65, 91)]
        digits = list(str(i) for i in range(10))

        self._tokens = special + chars + digits + ['<pad>', '<sos>']
        self._token2idx_dict = dict(
            (token, i) for i, token in enumerate(self._tokens))

        # every token must have an embedding
        self.n_embed = len(self._tokens)

        # <pad> and <sos> are not valid outputs
        self.n_output = len(self._tokens) - 2

    def idx2token(self, idx):
        return self._tokens[idx]

    def token2idx(self, token):
        return self._token2idx_dict[token]


class LRW1Dataset(VisionDataset):
    def __init__(self, root, splits, transform=None, easy=False, vocab=None):
        super().__init__(os.path.join(root, 'lipread_mp4'),
                         transform=transform)
        self.easy = easy
        self.vocab = vocab
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(splits, class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def _find_classes(self, root):
        classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])

        if self.easy:
            classes = classes[:50]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, splits, class_to_idx):
        if isinstance(splits, str):
            splits = [splits]

        assert all([x in ['train', 'val', 'test'] for x in splits])

        samples = []

        def is_valid_file(path):
            return path.lower().endswith('.mp4')

        for target in sorted(class_to_idx.keys()):
            for split in splits:
                d = os.path.join(self.root, target, split)

                assert os.path.isdir(d)

                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)

                        if is_valid_file(path):

                            if self.vocab is None:
                                # if no vocab then this is a
                                # classification task
                                label = torch.tensor(class_to_idx[target])
                            else:
                                # else look up indices and add sos + eos
                                label = torch.cat([
                                    torch.tensor(
                                        [self.vocab.token2idx('<sos>')]),
                                    torch.tensor([
                                        self.vocab.token2idx(token)
                                        for token in target
                                    ]),
                                    torch.tensor(
                                        [self.vocab.token2idx('<eos>')])
                                ])

                            samples.append((path, label))

        return samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        video = video_loader(path)

        assert len(video) == 29

        if self.transform is not None:
            video = self.transform(video)

        # squeeze channel dim
        video = video.squeeze(0)

        return video, target

    def __len__(self):
        return len(self.samples)


class PretrainLabel():
    def __init__(self, filename):
        self.all_utterances = self._preprocess(filename)

    def _preprocess(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()[4:]

            # Dictionary containing all possible utterances of certain length.
            # Length is the key and and defaultdict is used so there is no need
            # to take care of list creation for unseen keys
            all_utterances = defaultdict(list)

            prev_word_end = None

            for line in lines:
                word, start, end, __ = line.split()

                if prev_word_end != start:
                    # each word is represented by a list containing
                    # dicts for each word so not looking at any
                    # previous words corresponds to a list that
                    # contains an empty list
                    #
                    # this is done when when the end timestamp of the
                    # previous words does not equal to the start
                    # timestamp of the current word
                    possible_beginnings = [[]]

                w = {'word': word, 'start': float(start), 'end': float(end)}

                # get all possible utterances by prepending all possible
                # beginnings to the current word
                # note that we are using dicts that contain the start and end
                # because in the end we want to know the start and end of the
                # whole utterance
                utterances = [
                    beginning + [w] for beginning in possible_beginnings
                ]

                # append all utterances to the global dict
                for word_list in utterances:

                    # in the final list we only want the utterance and its
                    # start/end
                    utterance = {
                        'utterance': ' '.join([w['word'] for w in word_list]),
                        'start': word_list[0]['start'],
                        'end': word_list[-1]['end']
                    }

                    all_utterances[len(word_list)].append(utterance)

                # on next iteration we can append the word to any of the
                # utterances or start from scratch (non-contiguous case is
                # handled earlier in the loop)
                possible_beginnings = utterances + [[]]

                prev_word_end = end

            return all_utterances

    def sample(self, max_length, min_length=1):
        assert min_length <= max_length

        possible_lengths = [
            k for k in self.all_utterances.keys()
            if k <= max_length and k >= min_length
        ]

        assert len(possible_lengths) > 0, print(possible_lengths,
                                                self.all_utterances)
        utter_len = possible_lengths[randrange(0, len(possible_lengths))]

        # all utterances of length utter_len
        utterances = self.all_utterances[utter_len]

        # sampled utterance
        sampled = utterances[randrange(0, len(utterances))]

        return sampled['utterance'], (sampled['start'], sampled['end'])


# class LRS2PretrainWordDataset(VisionDataset):
#     def __init__(self, root, classes, vocab=None, transform=None):
#         super().__init__(root, transform=transform)
#         self.vocab = vocab
#         self.classes = classes
#         self.class_to_idx = {self.classes[i]: i for i in range(len(classes))}
#         self.samples = self._make_dataset(root)

#     def _make_dataset(self, root):
#         with open(os.path.join(root, 'pretrain.txt'), 'r') as f:
#             samples = []

#             def valid_length(start, end):
#                 clip_length = end - start
#                 return clip_length > 0.5 and clip_length < 2

#             for line in f.readlines():
#                 file_prefix = os.path.join(root, 'mvlrs_v1', 'pretrain',
#                                            line.strip())

#                 with open(file_prefix + '.txt', 'r') as f2:
#                     for line in f2.readlines()[4:]:
#                         word, start, end, _ = line.split()

#                         start, end = float(start), float(end)

#                         # skip if target is not in the list of given
#                         # words (500 LRW1 words) or if the clip is too
#                         # long or too short
#                         if word not in self.classes or not valid_length(
#                                 start, end):
#                             continue

#                         if self.vocab is None:
#                             target = torch.tensor(self.class_to_idx[word])
#                         else:
#                             target = torch.cat([
#                                 torch.tensor([self.vocab.token2idx('<sos>')]),
#                                 torch.tensor([
#                                     self.vocab.token2idx(token)
#                                     for token in word
#                                 ]),
#                                 torch.tensor([self.vocab.token2idx('<eos>')])
#                             ])

#                         if word in self.classes:
#                             sample = (file_prefix, target, (start, end))
#                             samples.append(sample)

#             return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         file_prefix, target, (start, end) = self.samples[idx]

#         video = video_loader(file_prefix + '.mp4', start, end)

#         if self.transform is not None:
#             video = self.transform(video)

#         video = video.squeeze(0)

#         return video, target


class LRS2PretrainDataset(VisionDataset):
    def __init__(self, root, vocab, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.min_seq_len = 2
        self.vocab = vocab

        # I guess we can assume that in e2e 128 batch and 25 frames
        # fit. When length is increased we can just interpolate that
        # and decrease batch as needed (and accumulate grads when
        # batchs get too small)

        self.valid_length = {
            # TODO: min is not needed
            1: (0.060, 0.640),
            2: (0.200, 0.990),
            3: (0.360, 1.320),
            4: (0.510, 1.630),
            5: (0.680, 1.940),
            6: (0.850, 2.240),
            7: (1.010, 2.530),
            8: (1.180, 2.820),
            9: (1.350, 3.100),
            10: (1.520, 3.380),
            11: (1.690, 3.650),
            12: (1.860, 3.920),
            13: (2.030, 4.180),
            14: (2.190, 4.450),
            15: (2.360, 4.710),
            16: (2.530, 4.980),
            17: (2.700, 5.240),
            18: (2.880, 5.510),
            19: (3.050, 5.780),
            20: (3.230, 6.050),
            21: (3.410, 6.303),
            22: (3.590, 6.578),
            23: (3.780, 6.810),
            24: (3.950, 7.070),
            25: (4.119, 7.250),
            26: (4.320, 7.520),
            27: (4.558, 7.822),
            28: (4.740, 8.086),
            29: (5.008, 8.460),
            30: (5.180, 8.770),
            31: (5.370, 9.148),
            32: (5.578, 9.573),
            33: (5.710, 9.818),
            34: (5.895, 10.193),
            35: (6.082, 10.572),
            36: (6.339, 10.965),
            37: (6.590, 11.344),
            38: (6.977, 11.718),
            39: (7.450, 12.130),
            40: (7.865, 12.310),
            41: (8.115, 12.071),
            42: (8.280, 11.110),
            43: (8.601, 10.811),
            44: (9.572, 11.070),
            45: (10.940, 11.238),
            46: (11.120, 11.503),
            47: (11.390, 11.676),
            48: (11.591, 11.953),
            49: (11.875, 12.163),
            50: (12.298, 12.442),
            51: (12.570, 12.570)
        }

        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        with open(os.path.join(root, 'pretrain.txt'), 'r') as f:
            samples = []

            for line in f.readlines():
                file_prefix = os.path.join(root, 'mvlrs_v1', 'pretrain',
                                           line.strip())

                label_path = Path(file_prefix + '.label')

                if not label_path.exists():
                    label = PretrainLabel(file_prefix + '.txt')
                    torch.save(label, str(label_path))
                else:
                    label = torch.load(str(label_path))

                # the loop in getitem can get stuck if all possible
                # sequences are too long. This makes sure we skip
                # videos that don't have any plausible candidates
                min_len = min([
                    x['end'] - x['start']
                    for x in label.all_utterances[self.min_seq_len]
                ]) if label.all_utterances[self.min_seq_len] else float('inf')

                # we want to exclude videos from dataset if
                # 1) the video does not have `self.min_seq_len` contiguous words
                # 2) length of the video is above the threshold
                if self.min_seq_len in label.all_utterances.keys(
                ) and min_len <= self.valid_length[self.max_seq_len][1]:
                    samples.append(file_prefix)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_prefix = self.samples[idx]

        label = torch.load(file_prefix + '.label')

        while True:
            # print('loop')
            # this is bit hacky and can lead to problems
            # but keep sampling until we get a sequence that is not too long
            utterance, (start, end) = label.sample(self.max_seq_len,
                                                   min_length=self.min_seq_len)

            if end - start <= self.valid_length[self.max_seq_len][1]:
                # print('break')
                break

        video = video_loader(file_prefix + '.mp4', start, end)

        if self.transform is not None:
            video = self.transform(video)

        target_indices = torch.tensor(
            [self.vocab.token2idx(char) for char in utterance])

        target = torch.cat([
            torch.tensor([self.vocab.token2idx('<sos>')]), target_indices,
            torch.tensor([self.vocab.token2idx('<eos>')])
        ])

        video = video.squeeze(0)

        return video, target


class LRS2FeatureDataset():
    def __init__(self, root, vocab, easy=False):
        self.max_seq_len = 2
        self.min_seq_len = 2
        self.vocab = vocab
        self.easy = easy
        self.samples = self._make_dataset(root)
        self.fps = 25

    def _make_dataset(self, root):
        with open(Path(root, 'pretrain.txt'), 'r') as f:
            samples = []

            lines = f.readlines() if not self.easy else f.readlines()[:500]

            for line in lines:
                path = Path(root, 'mvlrs_v1', 'pretrain_features',
                            line.strip() + '.pt')

                if path.exists():
                    features, label = torch.load(path)

                    if self.min_seq_len in label.all_utterances.keys():
                        samples.append(str(path))
                else:
                    print(f'{str(path)} not found')

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        features, label = torch.load(path)
        utterance, (start, end) = label.sample(self.max_seq_len,
                                               min_length=self.min_seq_len)

        first_frame = max(0, math.floor(start * self.fps))
        last_frame = min(features.shape[0], math.ceil(end * self.fps) + 1)
        features = features[first_frame:last_frame]

        target_indices = torch.tensor(
            [self.vocab.token2idx(char) for char in utterance])

        target = torch.cat([
            torch.tensor([self.vocab.token2idx('<sos>')]), target_indices,
            torch.tensor([self.vocab.token2idx('<eos>')])
        ])

        return features, target

    def increase_seq_len(self):
        self.max_seq_len += 1


class LRS2TestTrainDataset(VisionDataset):
    def __init__(self, test_or_train, root, vocab, transform=None, easy=False):
        super().__init__(root, transform=transform)
        assert test_or_train in ['train', 'test']
        self.train = test_or_train == 'train'
        self.vocab = vocab
        self.easy = easy
        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        filename = 'train.txt' if self.train else 'test.txt'

        with open(os.path.join(root, filename), 'r') as f:
            samples = []

            lines = f.readlines()[:100] if self.easy else f.readlines()

            for line in lines:
                path = line.split()[0].strip()
                mp4_path, txt_path = [
                    os.path.join(root, 'mvlrs_v1', 'main', path + suffix)
                    for suffix in ['.mp4', '.txt']
                ]

                with open(txt_path, 'r') as f:
                    text = f.readline().strip().split(maxsplit=1)[1]
                    indices = torch.tensor(
                        [self.vocab.token2idx(token) for token in text])
                    sample = (mp4_path, indices)

                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, indices = self.samples[idx]

        video = video_loader(path)

        if self.transform is not None:
            video = self.transform(video)

        video = video.squeeze(0)

        return video, indices


# class LRS2TextDataset(Dataset):
#     def __init__(self, root, vocab, subsets):
#         self.vocab = vocab
#         self.subsets = subsets
#         self.samples = self._make_dataset(root)

#     def _make_dataset(self, root):
#         samples = []

#         for subset in self.subsets:
#             with open(os.path.join(root, f'{subset}.txt'), 'r') as f:
#                 lines = f.readlines()

#                 for line in lines:
#                     subdir = 'pretrain' if subset == 'pretrain' else 'main'
#                     path = os.path.join(root, 'mvlrs_v1', subdir,
#                                         line.split()[0].strip() + '.txt')

#                     with open(path, 'r') as f:
#                         f_lines = f.readlines()
#                         text_lines = [
#                             l for l in f_lines if l.strip().startswith('Text:')
#                         ]
#                         assert len(text_lines) == 1

#                         sample = text_lines[0].split(maxsplit=1)[1].strip()
#                         indices = torch.cat([
#                             torch.tensor([self.vocab.token2idx('<sos>')]),
#                             torch.tensor([
#                                 self.vocab.token2idx(char) for char in sample
#                             ]),
#                             torch.tensor([self.vocab.token2idx('<eos>')])
#                         ])

#                     samples.append(indices)

#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# class LRS3FeatureDataset():
#     def __init__(self, root, vocab, easy=False):
#         self.max_seq_len = 1
#         self.easy = easy
#         self.vocab = vocab
#         self.samples = self._make_dataset(root)
#         self.fps = 25

#     def _make_dataset(self, root):
#         samples = []
#         pretrain_path = Path(root, 'pretrain_features')

#         for subdir in pretrain_path.iterdir() if not self.easy else list(
#                 pretrain_path.iterdir())[:20]:
#             for feature_path in subdir.glob('*.pt'):
#                 label_path = str(feature_path).replace('.pt', '.label')

#                 sample = (str(feature_path), label_path)
#                 samples.append(sample)

#         return samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         feature_path, label_path = self.samples[idx]

#         label = torch.load(label_path)
#         utterance, (start, end) = label.sample(self.max_seq_len)

#         features = torch.load(feature_path)
#         first_frame = max(0, math.floor(start * self.fps))
#         last_frame = min(features.shape[0], math.ceil(end * self.fps) + 1)
#         features = features[first_frame:last_frame]

#         target_indices = torch.tensor(
#             [self.vocab.token2idx(char) for char in utterance])

#         target = torch.cat([
#             torch.tensor([self.vocab.token2idx('<sos>')]), target_indices,
#             torch.tensor([self.vocab.token2idx('<eos>')])
#         ])

#         return features, target

# class PretrainFeatureDataset():
#     def __init__(self, root, vocab, easy=False):
#         self.lrs2 = LRS2FeatureDataset(os.path.join(root, 'lrs2'),
#                                        vocab,
#                                        easy=easy)
#         self.lrs3 = LRS3FeatureDataset(os.path.join(root, 'lrs3'),
#                                        vocab,
#                                        easy=easy)

#         assert self.lrs3.max_seq_len == self.lrs2.max_seq_len

#     def __len__(self):
#         return len(self.lrs2) + len(self.lrs3)

#     def __getitem__(self, idx):
#         if idx < len(self.lrs2):
#             value = self.lrs2[idx]
#         else:
#             value = self.lrs3[idx - len(self.lrs2)]

#         return value

#     def increase_seq_len(self):
#         self.lrs2.max_seq_len += 1
#         self.lrs3.max_seq_len += 1

#         assert self.lrs2.max_seq_len == self.lrs3.max_seq_len

#         return self.lrs2.max_seq_len


class LRS3PretrainDataset(VisionDataset):
    def __init__(self, root, vocab, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.min_seq_len = 2
        self.vocab = vocab

        self.valid_length = {
            1: (0.0700, 0.6800),
            2: (0.2100, 1.0400),
            3: (0.3800, 1.3700),
            4: (0.5500, 1.7000),
            5: (0.7200, 2.0200),
            6: (0.9000, 2.3300),
            7: (1.0800, 2.6300),
            8: (1.2500, 2.9300),
            9: (1.4300, 3.2200),
            10: (1.6000, 3.5000),
            11: (1.7800, 3.7800),
            12: (1.9500, 4.0500),
            13: (2.1200, 4.3200),
            14: (2.2900, 4.5900),
            15: (2.4600, 4.8600),
            16: (2.6300, 5.1200),
            17: (2.8000, 5.3900),
            18: (2.9600, 5.6600),
            19: (3.1200, 5.9300),
            20: (3.2800, 6.2000),
            21: (3.4600, 6.4600),
            22: (3.6300, 6.7300),
            23: (3.7900, 6.9850),
            24: (3.9500, 7.2350),
            25: (4.1100, 7.5300),
            26: (4.2500, 7.8200),
            27: (4.4060, 8.1300),
            28: (4.5400, 8.4400),
            29: (4.6650, 8.7800),
            30: (4.8100, 9.1300),
            31: (4.9275, 9.5725),
            32: (5.0250, 10.0100),
            33: (5.1825, 10.4325),
            34: (5.3080, 12.7720),
            35: (5.4300, 13.3650),
            36: (5.6250, 13.9420),
            37: (5.7230, 14.2835),
            38: (5.9370, 14.6565),
            39: (6.0520, 15.1300),
            40: (6.1590, 15.4310),
            41: (6.3500, 15.8800),
            42: (6.5025, 16.2900),
            43: (6.5900, 16.6300),
            44: (6.7400, 17.0140),
            45: (6.8370, 17.3620),
            46: (7.0065, 17.6775),
            47: (7.1715, 18.0200),
            48: (7.2585, 18.4990),
            49: (7.4490, 18.8510),
            50: (7.6210, 19.2280),
            51: (7.7790, 19.4240),
            52: (11.9260, 19.8025),
            53: (19.6550, 20.0800),
            54: (20.0540, 20.5540),
            55: (20.4550, 20.7030),
            56: (20.8940, 21.0470),
            57: (21.3840, 21.4560),
            58: (21.9600, 21.9600)
        }

        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        samples = []

        for f in Path(root, 'pretrain').rglob('**/*.txt'):
            file_prefix = str(f).replace('.txt', '')

            label_path = Path(file_prefix + '.label')

            if not label_path.exists():
                label = PretrainLabel(str(f))
                torch.save(label, str(label_path))
            else:
                label = torch.load(str(label_path))

            # the loop in getitem can get stuck if all possible
            # sequences are too long. This makes sure we skip
            # videos that don't have any plausible candidates
            min_len = min([
                x['end'] - x['start']
                for x in label.all_utterances[self.min_seq_len]
            ] if len(label.all_utterances[self.min_seq_len]) > 0 else
                          [float('inf')])

            if self.min_seq_len in label.all_utterances.keys(
            ) and min_len <= self.valid_length[self.max_seq_len][1]:
                samples.append(file_prefix)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_prefix = self.samples[idx]

        label = torch.load(file_prefix + '.label')

        while True:
            # print('loop2')
            # this is bit hacky and can lead to problems
            # but keep sampling until we get a sequence that is not too long
            utterance, (start, end) = label.sample(max_length=self.max_seq_len,
                                                   min_length=self.min_seq_len)

            if end - start <= self.valid_length[self.max_seq_len][1]:
                # print('break2')
                break

        video = video_loader(file_prefix + '.mp4', start, end)

        if self.transform is not None:
            video = self.transform(video)

        target_indices = torch.tensor(
            [self.vocab.token2idx(char) for char in utterance])

        target = torch.cat([
            torch.tensor([self.vocab.token2idx('<sos>')]), target_indices,
            torch.tensor([self.vocab.token2idx('<eos>')])
        ])

        video = video.squeeze(0)

        return video, target
