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
        lambda t: t.permute(0, 3, 1, 2),
        t.RandomFrameDrop(0.05), torch.unbind,
        t.Map(
            transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Grayscale()])),
        t.CenterCrop([122, 122]),
        t.RandomCrop([112, 112]),
        t.RandomHorizontalFlip(),
        t.Map(transforms.ToTensor()), torch.stack, lambda t: t.transpose(0, 1),
        t.Normalize()
    ])


def val_transform():
    return transforms.Compose([
        lambda t: t.permute(0, 3, 1, 2), torch.unbind,
        t.Map(
            transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Grayscale()])),
        t.CenterCrop([112, 112]),
        t.Map(transforms.ToTensor()), torch.stack, lambda t: t.transpose(0, 1),
        t.Normalize()
    ])


def pad_collate(samples, padding_value):
    """Add padding to samples to make them have equal length

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

        # transformer model must have embedding vector for each token (even for
        # pad even though it is never used)
        self.n_embed = len(self._tokens)

        # but when output probabilities for next token are produced predicting
        # pad or sos should not be possible
        self.n_output = len(self._tokens) - 2

    def idx2token(self, idx):
        return self._tokens[idx]

    def token2idx(self, token):
        return self._token2idx_dict[token]


class LRW1Dataset(VisionDataset):
    def __init__(self, root, subdir, transform=None, easy=False, vocab=None):
        super().__init__(os.path.join(root, 'lipread_mp4'),
                         transform=transform)
        self.easy = easy
        self.vocab = vocab
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(subdir, class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

    def _find_classes(self, root):
        classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])

        if self.easy:
            classes = classes[:50]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, subdir, class_to_idx):
        samples = []

        def is_valid_file(path):
            return path.lower().endswith('.mp4')

        for target in sorted(class_to_idx.keys()):
            d = os.path.join(self.root, target, subdir)

            assert os.path.isdir(d)

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):

                        if self.vocab is None:
                            # if no vocab then this is a classification task
                            label = class_to_idx[target]
                        else:
                            # else look up indices and add sos + eos
                            label = torch.cat([
                                torch.tensor([self.vocab.token2idx('<sos>')]),
                                torch.tensor([
                                    self.vocab.token2idx(token)
                                    for token in target
                                ]),
                                torch.tensor([self.vocab.token2idx('<eos>')])
                            ])

                        samples.append((path, label))

        return samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = video_loader(path)

        assert len(sample) == 29

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

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

        assert len(possible_lengths) > 0
        utter_len = possible_lengths[randrange(0, len(possible_lengths))]

        # all utterances of length utter_len
        utterances = self.all_utterances[utter_len]

        # sampled utterance
        sampled = utterances[randrange(0, len(utterances))]

        return sampled['utterance'], (sampled['start'], sampled['end'])


class LRS2PretrainDataset(VisionDataset):
    def __init__(self, root, vocab, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.vocab = vocab
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

                # some pretrain videos have no two consecutive words
                # so let's not include such videos in samples because
                # then we might sample a video that has less than 3
                # video frames => 3d conv fails
                if 2 in label.all_utterances.keys():
                    samples.append(file_prefix)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_prefix = self.samples[idx]

        label = torch.load(file_prefix + '.label')
        utterance, (start, end) = label.sample(self.max_seq_len, min_length=2)

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


# Validating seq2seq pretraining is difficult with the real validation
# data since pretraining is done with curriculum learning whereas
# validation sequences are fixed length (and word boundaries are not
# known). Therefore pretrain dataset is further spit into train and
# val subsets to test seq2seq architecture and hyperparams


class LRS2PretrainTrainSplit(LRS2PretrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self.samples[:int(0.8 * len(self.samples))]


class LRS2PretrainValSplit(LRS2PretrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self.samples[int(0.8 * len(self.samples)):]


class LRS2FeatureDataset():
    def __init__(self, root, vocab):
        self.max_seq_len = 1
        self.vocab = vocab
        self.samples = self._make_dataset(root)
        self.fps = 25

    def _make_dataset(self, root):
        with open(Path(root, 'pretrain.txt'), 'r') as f:
            samples = []

            for line in f.readlines():
                path = Path(root, 'mvlrs_v1', 'pretrain_features',
                            line.strip() + '.pt')

                if path.exists():
                    samples.append(str(path))
                else:
                    print(f'{str(path)} not found')

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        features, label = torch.load(path)
        utterance, (start, end) = label.sample(self.max_seq_len)

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


class LRS2FeatureValSplit(LRS2FeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self.samples[int(0.8 * len(self.samples)):]


class LRS2FeatureTrainSplit(LRS2FeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self.samples[:int(0.8 * len(self.samples))]


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


class LRS2TextDataset(Dataset):
    def __init__(self, root, vocab, subsets):
        self.vocab = vocab
        self.subsets = subsets
        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        samples = []

        for subset in self.subsets:
            with open(os.path.join(root, f'{subset}.txt'), 'r') as f:
                lines = f.readlines()

                for line in lines:
                    subdir = 'pretrain' if subset == 'pretrain' else 'main'
                    path = os.path.join(root, 'mvlrs_v1', subdir,
                                        line.split()[0].strip() + '.txt')

                    with open(path, 'r') as f:
                        f_lines = f.readlines()
                        text_lines = [
                            l for l in f_lines if l.strip().startswith('Text:')
                        ]
                        assert len(text_lines) == 1

                        sample = text_lines[0].split(maxsplit=1)[1].strip()
                        indices = torch.cat([
                            torch.tensor([self.vocab.token2idx('<sos>')]),
                            torch.tensor([
                                self.vocab.token2idx(char) for char in sample
                            ]),
                            torch.tensor([self.vocab.token2idx('<eos>')])
                        ])

                    samples.append(indices)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
