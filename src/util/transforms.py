import torch
from .videotransforms import video_transforms


class Normalize:
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std


class RandomFrameDrop:
    def __init__(self, p):
        self.p = p

    def __call__(self, tensor, padding='end'):
        t = tensor[torch.rand(len(tensor)) > self.p]

        if padding == 'end':
            padded = torch.zeros_like(tensor)
            padded[:len(t)] = t
            return padded

        return t


class CenterCrop():
    def __init__(self, size):
        self.t = video_transforms.CenterCrop(size)

    def __call__(self, clip):
        return self.t(clip)


class RandomCrop():
    def __init__(self, size):
        self.t = video_transforms.RandomCrop(size)

    def __call__(self, clip):
        return self.t(clip)


class RandomHorizontalFlip():
    def __init__(self):
        self.t = video_transforms.RandomHorizontalFlip()

    def __call__(self, clip):
        return self.t(clip)


class Map:
    def __init__(self, t):
        self.t = t

    def __call__(self, clip):
        return [self.t(frame) for frame in clip]
