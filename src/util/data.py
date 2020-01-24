import os
import torch
from random import randint
from collections import defaultdict
from torchvision.datasets.vision import VisionDataset
import torchvision
import util.transforms as t
import torchvision.transforms as transforms


def video_loader(path):
    video, audio, info = torchvision.io.read_video(path, pts_unit='sec')
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


def pad_collate(samples, padding_value, sos_value):
    x, y = zip(*samples)

    collated = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    collated_y = torch.nn.utils.rnn.pad_sequence(y,
                                                 batch_first=True,
                                                 padding_value=padding_value)

    # add sos to the beginning and pad to the end
    # TODO: would it be better to use <eos> and <pad> instead of just pad?
    batch_size = len(x)
    collated_y = torch.cat([
        torch.tensor([sos_value] * batch_size).view(batch_size, 1), collated_y,
        torch.tensor([padding_value] * batch_size).view(batch_size, 1)
    ],
                           dim=-1)

    return collated, collated_y


class LRS2PretrainSample():
    def __init__(self, path, text):
        self.path = path

        lines = text.split('\n')
        self.text = ' '.join(lines[0].split()[1:])
        self.conf = int(lines[1].split()[1])

        prev_end = -1
        possible_beginnings = [[]]

        # dict of continuous utterances where length of the utterance (words)
        # is the key
        self.all_utterances = defaultdict(list)

        for line in lines[4:]:
            if not line.strip():
                continue  # skip empty lines

            word, start, end, _ = line.split()

            if prev_end != start:
                possible_beginnings = [[]]

            utterances = [
                beginning + [{
                    'word': word,
                    'start': float(start),
                    'end': float(end)
                }] for beginning in possible_beginnings
            ]

            for u in utterances:
                self.all_utterances[len(u)].append({
                    'utterance': [e['word'] for e in u],
                    'start':
                    u[0]['start'],
                    'end':
                    u[-1]['end']
                })

            possible_beginnings = utterances + [[]]
            prev_end = end

    def sample(self, max_length):

        possible_lengths = [
            k for k in self.all_utterances.keys() if k <= max_length
        ]

        # sample len from possible lengths
        utter_len = possible_lengths[randint(0, len(possible_lengths) - 1)]

        # all utterances of length utter_len
        utterances = self.all_utterances[utter_len]

        # sampled utterance
        sampled = utterances[randint(0, len(utterances) - 1)]

        return sampled['utterance'], (sampled['start'], sampled['end'])


class CharVocab:
    def __init__(self, sos=False, blank=False):

        # vocab has 26 chars, 10 digits, space, apostophe and pad
        # + for TM-seq2seq [sos] and TM-CTC [blank]
        self.tokens = ['<pad>', ' ', "'"] + [chr(i)
                                             for i in range(65, 91)] + list(
                                                 str(i) for i in range(10))

        if sos:
            self.tokens.append('<sos>')

        if blank:
            self.tokens.append('<blank>')

        self.token2idx_dict = dict(
            (token, i) for i, token in enumerate(self.tokens))

    def idx2token(self, idx):
        return self.tokens[idx]

    def token2idx(self, token):
        return self.token2idx_dict[token]


class LRS2PretrainDataset(VisionDataset):
    def __init__(self, root, loader, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.loader = loader

        self.vocab = CharVocab(sos=True)
        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        with open(os.path.join(root, 'pretrain.txt'), 'r') as f:
            samples = []

            for line in f.readlines():
                mp4_path, txt_path = [
                    os.path.join(root, 'mvlrs_v1', 'pretrain',
                                 line.strip() + suffix)
                    for suffix in ['.mp4', '.txt']
                ]

                with open(txt_path, 'r') as f:
                    sample = LRS2PretrainSample(mp4_path, f.read())

                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        utterance, (start, end) = sample.sample(self.max_seq_len)

        video = self.loader(sample.path, start, end)

        if self.transform is not None:
            video = self.transform(video)

        return video, torch.tensor(
            [self.vocab.token2idx(char) for char in ' '.join(utterance)])


class LRW1Dataset(VisionDataset):
    def __init__(self,
                 root,
                 subdir,
                 loader,
                 transform=None,
                 easy=False,
                 classification=True):
        super().__init__(root, transform=transform)
        self.easy = easy
        self.classification = classification
        # TODO: params should not be hardcoded
        self.vocab = CharVocab(sos=True)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(root, subdir, class_to_idx)

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, root):
        classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])

        if self.easy:
            classes = classes[:50]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, rootdir, subdir, class_to_idx):
        samples = []

        def is_valid_file(path):
            return path.lower().endswith('.mp4')

        for target in sorted(class_to_idx.keys()):
            d = os.path.join(rootdir, target, subdir)

            if not os.path.isdir(d):
                print(f'DATA: {d} is not dir -- continue')
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):
                        target_class = class_to_idx[target]
                        target_chars = torch.tensor(
                            [self.vocab.token2idx(token) for token in target])
                        item = (path, target_class, target_chars)
                        samples.append(item)

        return samples

    def __getitem__(self, idx):
        path, target_class, target_chars = self.samples[idx]
        sample = self.loader(path)

        assert len(sample) == 29

        if self.transform is not None:
            sample = self.transform(sample)

        target = target_class if self.classification else target_chars

        return sample, target

    def __len__(self):
        return len(self.samples)
