import io
import os
import pickle

import numpy as np
import torch
from PIL import Image
from learn2learn.vision.datasets import TieredImagenet


unsampled_classes = []
sampled = []
cur_sample = []
support_xs_ids_sampled = {}

class TieredImageNet(TieredImagenet):
    def __init__(self, root, partition="train", mode='coarse', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        tiered_imaganet_path = os.path.join(self.root, 'tieredImageNet')
        short_partition = 'val' if partition == 'validation' else partition
        labels_path = os.path.join(tiered_imaganet_path, short_partition + '_labels.pkl')
        images_path = os.path.join(tiered_imaganet_path, short_partition + '_images_png.pkl')
        with open(images_path, 'rb') as images_file:
            self.images = pickle.load(images_file)
        with open(labels_path, 'rb') as labels_file:
            self.labels = pickle.load(labels_file)
            self.coarse2fine = {}
            for c, f in zip(self.labels['label_general'], self.labels['label_specific']):
                if c in self.coarse2fine:
                    if f not in self.coarse2fine[c]:
                        self.coarse2fine[c].append(f)
                else:
                    self.coarse2fine[c] = [f]
            if self.mode == 'coarse':
                self.labels = self.labels['label_general']
            elif self.mode == 'fine':
                self.labels = self.labels['label_specific']
            else:
                raise NotImplementedError

    @property
    def num_classes(self):
        return len(np.unique(self.labels))


class TieredImageNetTrainDataset(TieredImageNet):
    def __init__(self, args, partition='train', train_transform=None, session=0):
        super(TieredImageNetTrainDataset, self).__init__(
            root=args.tieredImageNet_root,
            partition=partition,
            mode='fine')
        self.n_ways = args.way
        self.n_shots = args.shot
        self.n_queries = args.query
        self.train_transform = train_transform
        self.upperbound = args.upperbound

        self.data = {}
        for idx in range(len(self.images)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.images[idx])
        self.classes = list(self.data.keys())

        self.session = session
        global unsampled_classes, sampled, cur_sample
        if session == 1:
            unsampled_classes = self.classes
        self.load_data = args.load_data
        assert len(unsampled_classes) != 0, 'no more new classes(；´д｀)ゞ'
        if len(unsampled_classes) > self.n_ways:
            cls_sampled = np.random.choice(unsampled_classes, self.n_ways, False)
            cur_sample = cls_sampled.tolist()
            sampled.append(cur_sample)
            unsampled_classes = np.setxor1d(unsampled_classes, cls_sampled)
        else:
            cls_sampled = np.array(unsampled_classes) if unsampled_classes is not np.ndarray else unsampled_classes
            cur_sample = cls_sampled.tolist()
            sampled.append(cur_sample)
            unsampled_classes = np.setxor1d(unsampled_classes, cls_sampled)
        self.new_class_num = len(cur_sample)

        support_xs = []
        support_ys = []
        cls_sampled = cur_sample
        if len(sampled) > 1:
            sampled_ = np.array([y for x in sampled[:-1] for y in x])  # [[],[],[]]->[]
            early = len(sampled_)
            if self.upperbound:
                self.new_class_num = early + len(cls_sampled)
                for idx, cls in enumerate(sampled_):
                    imgs = np.asarray(self.data[cls])
                    support_xs_ids = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
                    support_xs.append(imgs[support_xs_ids])
                    support_ys.append([len(self.coarse2fine) + idx] * self.n_shots)
        else:
            early = 0
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls])
            support_xs_ids_sampled[idx + early] = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled[idx + early]])
            support_ys.append([len(self.coarse2fine) + idx + early] * self.n_shots)

        support_xs, support_ys = np.array(support_xs), np.array(support_ys)
        support_xs = support_xs.reshape(-1)
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        support_xs = torch.stack(list(map(lambda x: self.train_transform(self._load_png_byte(x[0])), support_xs)))
        self.support_xs = support_xs
        self.support_ys = support_ys.reshape(-1)

    def __getitem__(self, item):
        support_xs = self.support_xs[item]
        support_ys = self.support_ys[item]
        return support_xs, support_ys

    def _load_png_byte(self, bytes):
        return Image.open(io.BytesIO(bytes))

    def __len__(self):
        return len(self.support_ys)

class TieredImageNetTestDataset(TieredImageNet):
    def __init__(self, args, partition='train', test_transform=None, session=0):
        super(TieredImageNetTestDataset, self).__init__(
            root=args.tieredImageNet_root,
            partition=partition,
            mode='fine')
        self.n_ways = args.way
        self.n_shots = args.shot
        self.n_queries = args.query
        self.test_transform = test_transform

        self.label2fine = {}

        self.data = {}
        for idx in range(len(self.images)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.images[idx])
        self.classes = list(self.data.keys())

        self.session = session
        self.load_data = args.load_data

        query_xs = []
        query_ys = []
        cls_sampled = cur_sample
        if len(sampled) > 1:
            sampled_ = np.array([y for x in sampled[:-1] for y in x])
            early = len(sampled_)
            for idx, cls in enumerate(sampled_):
                imgs = np.asarray(self.data[cls])
                query_xs_ids = np.random.choice(range(imgs.shape[0]), self.n_queries, False)
                query_xs.append(imgs[query_xs_ids])
                query_ys.append([len(self.coarse2fine) + idx] * query_xs_ids.shape[0])
                self.label2fine[len(self.coarse2fine) + idx] = cls
        else:
            early = 0

        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls])
            query_xs_ids = np.random.choice(np.setxor1d(range(imgs.shape[0]), support_xs_ids_sampled[idx + early]),
                                            self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([len(self.coarse2fine) + idx + early] * query_xs_ids.shape[0])
            self.label2fine[len(self.coarse2fine) + idx + early] = cls

        fine2coarse = {}
        coarse_id = {}
        coar = []
        for id in unsampled_classes:
            for coarse, fine in self.coarse2fine.items():
                if id in fine:
                    fine2coarse[id] = coarse
        assert len(fine2coarse) == len(unsampled_classes)
        for fine, coarse in fine2coarse.items():
            if coarse not in coar:
                coar.append(coarse)
                coarse_id[coarse] = len(coar) - 1

        for idx, cls in enumerate(np.array(unsampled_classes)):
            imgs = np.asarray(self.data[cls])
            query_xs_ids = np.random.choice(range(imgs.shape[0]), self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            id = fine2coarse[unsampled_classes[idx]]
            query_ys.append([id] * query_xs_ids.shape[0])
            if id in self.label2fine:
                self.label2fine[id].append(cls)
            else:
                self.label2fine[id] = [cls]

        query_xs, query_ys = np.array(query_xs), np.array(query_ys)
        num_ways, n_queries_per_way = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way))

        query_xs = query_xs.reshape((-1))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        query_xs = torch.stack(list(map(lambda x: self.test_transform(self._load_png_byte(x[0])), query_xs)))

        self.query_xs = query_xs
        self.query_ys = query_ys

    def __getitem__(self, item):
        query_xs = self.query_xs[item]
        query_ys = self.query_ys[item]
        return query_xs, query_ys

    def _load_png_byte(self, bytes):
        return Image.open(io.BytesIO(bytes))

    def __len__(self):
        return len(self.query_ys)


