from PIL import Image
import os
import os.path
import numpy as np
import pickle
import random
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR100(VisionDataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, index=None, base_sess=None, mode='fine', args=None):

        super(CIFAR100, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.args = args
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.transform = transform

        self.data = []
        self.coarse_targets = []
        self.fine_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.fine_targets.extend(entry['labels'])
                else:
                    self.coarse_targets.extend(entry['coarse_labels'])
                    self.fine_targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.coarse_targets = np.asarray(self.coarse_targets)
        self.fine_targets = np.asarray(self.fine_targets)

        if base_sess is True:
            if mode == 'coarse':
                self.data, self.targets = self.SelectfromDefault(self.data, self.coarse_targets, index)
            elif mode == 'fine':
                self.data, self.targets = self.SelectfromDefault(self.data, self.fine_targets, index)
        else:  # new Class session
            if train is True:
                self.data, self.targets = self.NewClassSelector(self.data, self.fine_targets, index)
            else:
                self.data, self.targets = self.Newclasstest(self.data, self.coarse_targets, self.fine_targets, index)

        self._load_meta()

    def SelectfromDefault(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp

    def Newclasstest(self, data, coarse_targets, fine_targets, index):
        data_tmp = []
        targets_tmp = []
        now_fine = len(index)
        now_coarse = self.args.fine_class - now_fine
        for i in index:
            ind_cl = np.where(i == fine_targets)[0]
            new_cl = ind_cl
            selected_img = np.random.choice(new_cl, self.args.query, False)
            if data_tmp == []:
                data_tmp = data[selected_img]
                targets_tmp = fine_targets[selected_img] + [self.args.base_class]
            else:
                data_tmp = np.vstack((data_tmp, data[selected_img]))
                new_targets = fine_targets[selected_img] + [self.args.base_class]
                targets_tmp = np.hstack((targets_tmp, new_targets))
        if now_coarse != 0:
            for j in range(now_coarse):
                ind_cl = np.where((j + now_fine) == fine_targets)[0]
                new_cl = ind_cl
                selected_img = np.random.choice(new_cl, self.args.query, False)
                data_tmp = np.vstack((data_tmp, data[selected_img]))
                targets_tmp = np.hstack((targets_tmp, coarse_targets[selected_img]))
        return data_tmp, targets_tmp

    def NewClassSelector(self, data, targets, class_index):
        data_tmp = []
        targets_tmp = []
        index = []
        for i in class_index:
            ind_cl = np.where(i == targets)[0]
            ind_cl = ind_cl.tolist()
            new_cl = random.sample(ind_cl, self.args.shot)
            if index == []:
                index = np.array(new_cl)
            else:
                index = np.hstack((index, np.array(new_cl)))
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list)
        index = ind_np.reshape((len(ind_np), 1))
        for i in index:
            ind_cl = i
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl] + [self.args.base_class]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl] + [self.args.base_class]))

        return data_tmp, targets_tmp

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
