import os
from torchvision.datasets.vision import VisionDataset


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
