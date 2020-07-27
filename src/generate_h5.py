import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import data
from utils import transforms as t

Path.ls = lambda x: list(x.iterdir())

p = Path('/scratch/elec/puhe/c/LRS3-TED/lrw1/lipread_mp4/')

class_to_idx = dict(map(reversed, enumerate(sorted([x.name for x in p.ls()]))))

tr = transforms.Compose(
    [t.Permute(),
     t.GrayScale(),
     t.Crop(size=(112, 112), crop='center')])

dirs = sorted(p.iterdir())
N = sum(1 for d in dirs for _ in (d / 'train').glob('*.mp4'))

with h5py.File('/scratch/elec/puhe/c/LRS3-TED/lrw1/test.hdf5', 'a') as f:
    train = f.create_group('train')
    ds = train.create_dataset('data', shape=(N, 29, 1, 112, 112), dtype='int8')
    labels = train.create_dataset('labels', shape=(N, 1), dtype='int8')

    for i, d in tqdm(enumerate(dirs)):
        label = d.name

        for j, video_fp in enumerate(sorted((d / 'train').glob('*.mp4'))):
            x = tr(data.video_loader(video_fp)).numpy()

            ds[i * len(class_to_idx) + j] = x
            labels[i * len(class_to_idx) + j] = class_to_idx[label]
