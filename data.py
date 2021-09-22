import torch

from multiprocessing import cpu_count
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

def get_data(tasks, batch_size):
    transforms = [Compose([ToTensor(), Flatten(), Permute(torch.randperm(784))]) for _ in range(tasks)]

    train_datasets = [MNIST("inputs", train=True, download=True, transform=transform) for transform in transforms]
    test_datasets = [MNIST("inputs", train=False, download=True, transform=transform) for transform in transforms]

    train_loaders = [DataLoader(train_datasets[task], shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=cpu_count()) for task in range(tasks)]
    test_loaders = [DataLoader(test_datasets[task], shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=cpu_count()) for task in range(tasks)]

    data_size = [len(train_datasets[task]) for task in range(tasks)]
    return train_loaders, test_loaders, data_size

class Permute(object):
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        return sample[self.permutation]

class Flatten(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample.flatten()