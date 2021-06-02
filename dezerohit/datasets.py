import os
import gzip
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt

# from dezerohit.utils import get_file, cache_dir
# from dezerohit.transforms import Compose, Flatten, ToFloat, Normalize


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return (
                self.transform(self.data[index]),
                self.target_transform(self.label[index]),
            )

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

