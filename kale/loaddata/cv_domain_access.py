"""
Digits dataset (source and target domain) loading for MNIST, SVHN, MNIST-M (modified MNIST), and USPS. The code is based on 
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

import os
import sys
from kale.loaddata.dataset_access import DatasetAccess
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class PACSAccess(DatasetAccess):

    def __init__(self, data_path, domain='art_painting', transform='default', test_size=0.2):
        super().__init__(n_classes=7)
        if not os.path.exists(data_path):
            print('Data path \'%s\' does not' % data_path)
            sys.exit()
        if domain.lower() not in ['art_painting', 'cartoon', 'photo', 'sketch']:
            print('Invalid domain')
            sys.exit()
        self._domain = domain.lower()
        self._data_path = os.path.join(data_path, self._domain)
        if transform == 'default':
            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self._transform = transform
        self._dataset = ImageFolder(data_path, transform=self._transform)

        n_sample = len(self._dataset.imgs)
        n_test = int(n_sample * test_size)
        n_train = n_sample - n_test
        self.train, self.test = random_split(self._dataset, [n_train, n_test])

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test
