import os
import torch
from random import randint
from collections import defaultdict
from torchvision.datasets.vision import VisionDataset
import torchvision
import util.transforms as t
import torchvision.transforms as transforms


def video_loader(path, start_pts=0, end_pts=None):
    video, audio, info = torchvision.io.read_video(path,
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


def pad_collate(samples, padding_value, sos_value):
    x, y = zip(*samples)

    # x is a list of 1(=channel) x batch x 112 x 112
    # but pad_sequence only works if the first batch is the seq len
    x = [x_.squeeze(0) for x_ in x]

    collated = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)

    # add back the channel dimension
    collated = collated.unsqueeze(1)

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


class CharVocab:
    def __init__(self, sos=False, blank=False):

        # vocab has 26 chars, 10 digits, space, apostophe and pad
        # + <sos> for seq2seq and <blank> for CTC-based
        self._tokens = ['<pad>', ' ', "'"] + [chr(i)
                                              for i in range(65, 91)] + list(
                                                  str(i) for i in range(10))

        if sos:
            self._tokens.append('<sos>')

        if blank:
            self._tokens.append('<blank>')

        self._token2idx_dict = dict(
            (token, i) for i, token in enumerate(self._tokens))

    def idx2token(self, idx):
        return self._tokens[idx]

    def token2idx(self, token):
        return self._token2idx_dict[token]

    def __len__(self):
        return len(self._tokens)


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


class LRS2PretrainDataset(VisionDataset):
    def __init__(self, root, loader, vocab, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.loader = loader

        self.vocab = vocab
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


class LRS2TestTrainDataset(VisionDataset):
    def __init__(self, test_or_train, root, loader, vocab, transform=None):
        super().__init__(root, transform=transform)
        assert test_or_train in ['train', 'test']
        self.train = test_or_train == 'train'
        self.loader = loader
        self.vocab = vocab
        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        filename = 'train.txt' if self.train else 'test.txt'

        with open(os.path.join(root, filename), 'r') as f:
            samples = []

            for line in f.readlines():
                path = line.split()[0].strip()
                mp4_path, txt_path = [
                    os.path.join(root, 'mvlrs_v1', 'main', path + suffix)
                    for suffix in ['.mp4', '.txt']
                ]

                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    text = ' '.join(lines[0].split()[1:])
                    indices = torch.tensor(
                        [self.vocab.token2idx(token) for token in text])
                    sample = (mp4_path, indices)

                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, indices = self.samples[idx]

        video = self.loader(path)

        if self.transform is not None:
            video = self.transform(video)

        return video, indices


class LRW1Dataset(VisionDataset):
    def __init__(self,
                 root,
                 subdir,
                 loader,
                 transform=None,
                 easy=False,
                 vocab=None):
        super().__init__(root, transform=transform)
        self.easy = easy
        self.vocab = vocab
        self.loader = loader
        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(root, subdir, class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

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

            assert os.path.isdir(d)

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)

                    if is_valid_file(path):

                        if self.vocab is None:
                            # if no vocab then this is a classification task
                            label = class_to_idx[target]
                        else:
                            # else look up char indices from vocab
                            label = torch.tensor([
                                self.vocab.token2idx(token) for token in target
                            ])

                        samples.append((path, label))

        return samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)

        assert len(sample) == 29

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
