import torch


class Normalize:
    def __call__(self, tensor):
        return (tensor - tensor.mean()) / tensor.std()


class GrayScale:
    def __call__(self, tensor):
        # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        ret = (0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] +
               0.114 * tensor[:, 2]).unsqueeze(1)
        return ret / 255


class RandomFrameDrop:
    def __init__(self, p):
        self.p = p

    def __call__(self, tensor):
        t = tensor[torch.rand(len(tensor)) > self.p]

        padded = torch.zeros_like(tensor)
        padded[:len(t)] = t
        return padded


class Crop():
    def __init__(self, size, crop):
        assert crop in ['center', 'random']
        self.crop = crop
        self.size = size

    def __call__(self, video):
        w, h = self.size

        video_w, video_h = video.shape[2:]

        assert w == h
        assert video_w == video_h
        assert w % 2 == 0
        assert video_w % 2 == 0

        center_w = video_w // 2
        center_h = video_h // 2

        if self.crop == 'random':
            center_w += torch.randint(low=-10, high=10, size=(1, ))
            center_h += torch.randint(low=-10, high=10, size=(1, ))

        d = w // 2

        return video[:, :, center_h - d:center_h + d, center_w - d:center_w +
                     d]


class RandomHorizontalFlip():
    def __call__(self, video):
        if torch.rand(1).item() > 0.5:
            video = torch.flip(video, dims=[3])

        return video


class Permute:
    def __call__(self, x):
        # permute to (depth, channels, height, width)
        return x.permute(0, 3, 1, 2).contiguous()


class Transpose:
    def __call__(self, x):
        # tranpose to (channels, depth, height, width)
        return x.transpose(0, 1).contiguous()
