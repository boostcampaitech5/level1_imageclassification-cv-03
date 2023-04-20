import torchvision.transforms as T
from PIL import Image
import torch

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.CenterCrop(384),
            T.Resize(resize, Image.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class FlipAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class RotateAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class NoiseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class ColorAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.RandomPosterize(bits=1, p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class ContrastAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.ColorJitter(contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class SharpnessAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.RandomAdjustSharpness(sharpness_factor=10, p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class GrayscaleAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.Resize(resize, Image.BILINEAR),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=(0.532, 0.532, 0.532), std=(0.590, 0.590, 0.590)),
        ])

    def __call__(self, image):
        return self.transform(image)

class CenterCropAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = T.Compose([
            T.CenterCrop(size=384),
            T.Resize(resize, Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)