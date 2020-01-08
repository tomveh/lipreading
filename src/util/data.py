import os
from random import randint
from collections import defaultdict
from torchvision.datasets.vision import VisionDataset


class LRS2PretrainSample():
    def __init__(self, path, text):
        self.path = path

        lines = text.split('\n')
        # self.text = ' '.join(lines[0].split()[1:])
        # self.conf = int(lines[1].split()[1])

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
    def __init__(self, root, loader, transform=None):
        super().__init__(root, transform=transform)
        self.max_seq_len = 2
        self.loader = loader
        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        with open(os.path.join(root, 'pretrain.txt'), 'r') as f:
            samples = []

            # TODO: create vocabulary here

            for line in f.readlines()[:100]:
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

        return video, utterance


class LRW1Dataset(VisionDataset):
    def __init__(self, root, subdir, loader, transform=None):
        super().__init__(root, transform=transform)

        classes, class_to_idx = self._find_classes(self.root)
        samples = self._make_dataset(root, subdir, class_to_idx)

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, root):
        classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
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
                        item = (path, class_to_idx[target])
                        samples.append(item)

        return samples

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)

        try:
            assert len(sample) == 29
        except AssertionError:
            print(f'len of {path} is {len(sample)}')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
