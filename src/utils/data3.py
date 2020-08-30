import os
import tokenizers
from collections import defaultdict
from pathlib import Path
from random import randrange

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset

import utils.transforms as t
from tokenizers import CharBPETokenizer


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



def pad_collate(samples, tokenizer=None):
    """Pad data with zeros and tokenize targets if tokenizer is provided

    :param samples: list of tuples (x, y) where x is the video tensor and y is
    either scalar tensor (class) or target string
    :param tokenizer: tokenizer used to process the target string
    :returns: tuple (padded_data, padded_target)

    """
    data, targets = zip(*samples)

    padded_data = torch.nn.utils.rnn.pad_sequence(data,
                                               batch_first=True,
                                               padding_value=0)

    if tokenizer:
        targets = [batch.ids for batch in tokenizer.encode_batch(targets)]

    return padded_data, torch.stack(targets)


class SubwordTokenizer():
    def __init__(self, path):
        self.tokenizer = tokenizers.Tokenizer.from_file(path)
        self._idx_to_internal = dict(enumerate([idx for token, idx in self.tokenizer.get_vocab().items() if token not in ['<sos>', '<eos>']]))
        self._internal_to_idx = dict(map(reversed, self._idx_to_internal.items()))

    def get_output_vocab_size(self):
        # do not include pad and sos
        return len(self._idx_to_internal)

    def id_to_token(self, idx):
        pass


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


class PretrainDataset(VisionDataset):
    def __init__(self, root, vocab, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.min_seq_len = 2
        self.vocab = vocab

        # map from max_seq_len to maximum length of video in seconds
        # TODO: this is set by the scheduler
        self.max_lengths = dict([(2, 1), (3, 1.4), (4, 1.7), (5, 2), (6, 2.3),
                                 (7, 2.6), (8, 2.9), (9, 3.2), (10, 3.5),
                                 (11, 3.8), (12, 4), (13, 4.3), (14, 4.6),
                                 (15, 4.8), (16, 5.1), (17, 5.4), (18, 5.6),
                                 (19, 5.9), (20, 6.2), (21, 6.5), (22, 6.7),
                                 (23, 7), (24, 7.2), (25, 7.5), (26, 7.8),
                                 (27, 8.1), (28, 8.4), (29, 8.8), (30, 9.1),
                                 (31, 9.5), (32, 10), (33, 10.4), (34, 12.7),
                                 (35, 13.3), (36, 13.9), (37, 14), (38, 14),
                                 (39, 14), (40, 14), (41, 14), (42, 14),
                                 (43, 14), (44, 14), (45, 14), (46, 14),
                                 (47, 14), (48, 14), (49, 14), (50, 14),
                                 (51, 14), (52, 14), (53, 14), (54, 14),
                                 (55, 14), (56, 14), (57, 14), (58, 14)])

    def _make_dataset(self, root):
        samples = []

        for file_prefix in self.file_prefixes:
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
            # 1) the video does not have `self.min_seq_len` contiguous words or
            # 2) length of the video is above the manually set threshold
            if self.min_seq_len in label.all_utterances.keys(
            ) and min_len <= self.max_lengths[self.max_seq_len]:
                samples.append(file_prefix)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_prefix = self.samples[idx]

        label = torch.load(file_prefix + '.label')

        # keep sampling consecutive words until the length is less than max
        # length
        while True:
            utterance, (start, end) = label.sample(self.max_seq_len,
                                                   min_length=self.min_seq_len)
            if end - start <= self.max_lengths[self.max_seq_len]:
                break

        video = video_loader(file_prefix + '.mp4', start, end)

        if self.transform is not None:
            video = self.transform(video)

        video = video.squeeze(0)

        target = '<sos>' + utterance + '<eos>'

        return video, target


class LRS2PretrainDataset(PretrainDataset):
    def __init__(self, root, vocab, transform=None):
        super().__init__(root, vocab, transform=transform)
        self.file_prefixes = self._get_prefixes()
        self.samples = self._make_dataset(root)

    def _get_prefixes(self):
        with open(os.path.join(self.root, 'pretrain.txt'), 'r') as f:
            return [
                os.path.join(self.root, 'mvlrs_v1', 'pretrain', line.strip())
                for line in f.readlines()
            ]


class LRS3PretrainDataset(PretrainDataset):
    def __init__(self, root, vocab, transform=None):
        super().__init__(root, vocab, transform=transform)
        self.file_prefixes = self._get_prefixes()
        self.samples = self._make_dataset(root)

    def _get_prefixes(self):
        return [
            str(p).replace('.txt', '')
            for p in Path(self.root, 'pretrain').rglob('*.txt')
        ]
