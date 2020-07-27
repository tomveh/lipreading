import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

import utils.data as data
from models.frontend import VisualFrontend
from pretrain import VisualPretrainModule


class LRS2Dataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        with open(os.path.join(self.root, 'pretrain.txt'), 'r') as f:
            samples = []
            for line in f.readlines():
                file_prefix = os.path.join(self.root, 'mvlrs_v1', 'pretrain',
                                           line.strip())

                samples.append(file_prefix)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_prefix = self.samples[idx]
        video = data.video_loader(file_prefix + '.mp4')

        if self.transform:
            video = self.transform(video)

        return video, file_prefix


class LRS3Dataset():
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.samples = self._make_dataset(root)

    def _make_dataset(self, root):
        samples = []
        pretrain_path = Path(root) / 'pretrain'

        for subdir in pretrain_path.iterdir():
            file_prefixes = [
                str(path.parent / path.name.split('.')[0])
                for path in subdir.iterdir() if path.name.endswith('mp4')
            ]

            samples.extend(file_prefixes)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_prefix = self.samples[idx]
        video = data.video_loader(file_prefix + '.mp4')

        if self.transform:
            video = self.transform(video)

        return video, file_prefix


def generate_features(root, model_path=None, device='cuda'):
    transform = data.val_transform()

    if root.name.split('/')[-1] == 'lrs2':
        ds = LRS2Dataset(root, transform)
    elif root.name.split('/')[-1] == 'lrs3':
        ds = LRS3Dataset(root, transform)
    else:
        raise RuntimeError('Invalid dataset root', root)

    model = VisualFrontend().to(device)

    if model_path:
        if model_path.endswith('.pt'):
            model.load_state_dict(torch.load(model_path))
        elif model_path.endswith('.ckpt'):
            module = VisualPretrainModule.load_from_checkpoint(model_path)
            model.load_state_dict(module.model.frontend.state_dict())
        else:
            raise RuntimeError('unknown model_path filetype',
                               model_path.split('.')[-1])

        print('loaded model from', model_path)

    model.eval()

    dl = DataLoader(ds, batch_size=1, num_workers=12)

    with torch.no_grad():
        for video, file_prefix in tqdm.tqdm(dl):
            video, file_prefix = video.to(device), file_prefix[0]

            save_path = file_prefix.replace('pretrain',
                                            'pretrain_features') + '.pt'

            if video.shape[2] > 2500:
                # skip sequences that are too long to fit in the gpu memory
                print('skipping', file_prefix, video.shape)
                continue

            # features are now [seq_len, d_model]
            features = model(video).squeeze(0)
            label = data.PretrainLabel(file_prefix + '.txt')

            # create parent directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            obj = (features.to('cpu'), label)

            torch.save(obj, save_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    default_root = '/work/t405/T40511/work/vehvilt2/'
    parser.add_argument('--root', default=default_root, type=str)

    default_model_path = '/u/46/vehvilt2/unix/lipreading/lightning_logs/pretrain/version_3/weights/frontend_weights.pt'
    parser.add_argument('--model_path', default=default_model_path, type=str)

    parser.add_argument('--ds', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)

    params = parser.parse_args()

    root = Path(params.root, params.ds)

    generate_features(root, model_path=params.model_path, device=params.device)
