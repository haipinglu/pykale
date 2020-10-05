"""
Digits dataset (source and target domain) loading for MNIST, SVHN, MNIST-M (modified MNIST), and USPS. The code is based on 
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

import os
import sys
from kale.loaddata.dataset_access import DatasetAccess
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch


transform_default = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


class PACSAccess(DatasetAccess):

    def __init__(self, data_path, domain='art_painting', transform='default',
                 test_size=0.2, random_state=144):
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
            # self._transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(
            #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #     ),
            # ])
            self._transform = transform_default
        else:
            self._transform = transform
        self._dataset = ImageFolder(self._data_path, transform=self._transform)

        torch.manual_seed(random_state)
        n_sample = len(self._dataset.imgs)
        n_test = int(n_sample * test_size)
        n_train = n_sample - n_test
        self.train, self.test = random_split(self._dataset, [n_train, n_test])

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test


class VLCSAccess(DatasetAccess):

    def __init__(self, data_path, domain='CALTECH', transform='default'):

        super().__init__(n_class=5)
        if not os.path.exists(data_path):
            print('Data path \'%s\' does not' % data_path)
            sys.exit()
        self.data_path = data_path
        if domain.upper() not in ['CALTECH', 'LABELME', 'PASSCAL', 'SUN']:
            print('Invalid domain')
            sys.exit()
        self._domain = domain.upper()
        self._data_path = os.path.join(data_path, self._domain)
        if transform == 'default':
            self._transform = transform_default
        else:
            self._transform = transform
        # self._dataset = ImageFolder(data_path, transform=self._transform)
    
    def get_train(self):
        train_path = os.path.join(self._data_path, 'full')
        self.train = ImageFolder(train_path, transform=self._transform)
        return self.train

    def get_test(self):

        test_path = os.path.join(self._data_path, 'test')
        self.test = ImageFolder(test_path, transform=self._transform)

        return self.test



class MultiAccess(DatasetAccess):

    def __init__(self, data_path, data_name, domains, transform='default', **kwargs):
        # super().__init__(n_classes=7)
        domain_info = {'pacs': (['art_painting', 'cartoon', 'photo', 'sketch'], 
                                PACSAccess, 7),
                       'vlcs': (['CALTECH', 'LABELME', 'PASSCAL', 'SUN'], 
                                VLCSAccess, 5)}
        domain_list, data_access, _n_class = domain_info[data_name]
        super().__init__(n_classes=_n_class)
        self.data_ = dict()
        for d in domain_list:
            if d.lower() not in domain_list:
                print('Invalid target domain')
                sys.exit()
            self.data_[d] = data_access(data_path, domain=d, transform=transform, **kwargs)
                                        #  test_size=test_size, random_state=random_state)

    def get_train(self):
        train_list = []
        for key in self.data_:
            train_list.append(self.data_[key].get_train())
        self.train = ConcatDataset(train_list)

        return self.train
        
    def get_test(self):
        test_list = []
        for key in self.data_:
            test_list.append(self.data_[key].get_test())
        self.test = ConcatDataset(test_list)
        
        return self.test
