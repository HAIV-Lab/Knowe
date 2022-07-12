import os

import numpy as np
import torch
from torch.utils.data import Dataset

from robustness.tools import folder
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness.tools.helpers import get_label_mapping

unsampled_classes = []
sampled = []
cur_sample = []
support_xs_ids_sampled = {}


class BREEDSFactory:
    def __init__(self, info_dir, data_dir):
        self.info_dir = info_dir
        self.data_dir = data_dir

    def get_breeds(self, ds_name, partition, mode='coarse', transforms=None, split=None):
        superclasses, subclass_split, label_map = self.get_classes(ds_name, split)
        partition = 'val' if partition == 'validation' else partition  # train or val
        print(f"==> Preparing dataset {ds_name}, mode: {mode}, partition: {partition}..")
        if split is not None:
            index = 0 if partition == 'train' else 1
            return self.create_dataset(partition, mode, subclass_split[index], transforms)
        else:
            return self.create_dataset(partition, mode, subclass_split[0], transforms)

    def create_dataset(self, partition, mode, subclass_split, transforms):
        coarse_custom_label_mapping = get_label_mapping("custom_imagenet", subclass_split)
        fine_subclass_split = [[item] for sublist in subclass_split for item in sublist]
        fine_custom_label_mapping = get_label_mapping("custom_imagenet", fine_subclass_split)
        if mode == 'coarse':
            active_custom_label_mapping = coarse_custom_label_mapping
            active_subclass_split = subclass_split
        elif mode == 'fine':
            active_custom_label_mapping = fine_custom_label_mapping
            active_subclass_split = fine_subclass_split
        else:
            raise NotImplementedError
        dataset = folder.ImageFolder(root=os.path.join(self.data_dir, partition), transform=transforms,
                                     label_mapping=active_custom_label_mapping)
        coarse2fine = self.extract_c2f_from_dataset(dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition)
        setattr(dataset, 'num_classes', len(active_subclass_split))
        setattr(dataset, 'coarse2fine', coarse2fine)
        return dataset

    def extract_c2f_from_dataset(self, dataset,coarse_custom_label_mapping,fine_custom_label_mapping,partition):
        classes, original_classes_to_idx = dataset._find_classes(os.path.join(self.data_dir, partition))
        _, coarse_classes_to_idx = coarse_custom_label_mapping(classes, original_classes_to_idx)
        _, fine_classes_to_idx = fine_custom_label_mapping(classes, original_classes_to_idx)
        coarse2fine = {}
        for k, v in coarse_classes_to_idx.items():
            if v in coarse2fine:
                coarse2fine[v].append(fine_classes_to_idx[k])
            else:
                coarse2fine[v] = [fine_classes_to_idx[k]]
        return coarse2fine

    def get_classes(self, ds_name, split=None):
        if ds_name == 'living17':
            return make_living17(self.info_dir, split)
        elif ds_name == 'entity30':
            return make_entity30(self.info_dir, split)
        elif ds_name == 'entity13':
            return make_entity13(self.info_dir, split)
        elif ds_name == 'nonliving26':
            return make_nonliving26(self.info_dir, split)
        else:
            raise NotImplementedError

class BreedsSessionTrainDataset(Dataset):
    def __init__(self, args, dataset, train_transform=None, session=0):
        super(Dataset, self).__init__()
        self.n_ways = args.way
        self.n_shots = args.shot
        self.n_queries = args.query
        self.train_transform = train_transform
        self.data = {}
        self.loader = dataset.loader
        self.coarse2fine = dataset.coarse2fine
        self.upperbound = args.upperbound

        if hasattr(dataset, "samples"):
            self.images = [s[0] for s in dataset.samples]
        elif hasattr(dataset, "images"):
            self.images = dataset.images
        if hasattr(dataset, "targets"):
            self.labels = dataset.targets
        elif hasattr(dataset, "labels"):
            self.labels = dataset.labels

        for idx in range(len(self.images)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.images[idx])
        self.classes = sorted(list(self.data.keys()))

        self.session = session
        global unsampled_classes, sampled, cur_sample
        if session == 1:
            unsampled_classes = self.classes

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
            sampled_ = np.array([y for x in sampled[:-1] for y in x])
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
            support_xs_ids_sampled[idx + early] = np.random.choice(range(imgs.shape[0]), self.n_shots,
                                                                   False)
            support_xs.append(imgs[support_xs_ids_sampled[idx + early]])
            support_ys.append([len(self.coarse2fine) + idx + early] * self.n_shots)

        support_xs, support_ys = np.array(support_xs), np.array(support_ys)
        support_xs = support_xs.reshape((-1, 1))

        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        support_xs = torch.stack(
            list(map(lambda x: self.train_transform(self.loader(x.squeeze(0)[0])), support_xs)))
        self.support_xs = support_xs
        self.support_ys = support_ys.reshape(-1)

    def __getitem__(self, item):
        support_xs = self.support_xs[item]
        support_ys = self.support_ys[item]
        return support_xs, support_ys

    def __len__(self):
        return len(self.support_ys)

class BreedsSessionTestDataset(Dataset):
    def __init__(self, args, dataset, test_transform=None, session=0):
        super(Dataset, self).__init__()
        self.n_ways = args.way
        self.n_shots = args.shot
        self.n_queries = args.query
        self.test_transform = test_transform
        self.data = {}
        self.loader = dataset.loader
        self.coarse2fine = dataset.coarse2fine

        self.label2fine = {}

        if hasattr(dataset, "samples"):
            self.images = [s[0] for s in dataset.samples]
        elif hasattr(dataset, "images"):
            self.images = dataset.images
        if hasattr(dataset, "targets"):
            self.labels = dataset.targets
        elif hasattr(dataset, "labels"):
            self.labels = dataset.labels

        for idx in range(len(self.images)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.images[idx])
        self.classes = sorted(list(self.data.keys()))

        self.session = session

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
            query_xs_ids = np.random.choice(range(imgs.shape[0]), self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([len(self.coarse2fine) + idx + early] * query_xs_ids.shape[0])
            self.label2fine[len(self.coarse2fine) + idx + early] = cls

        fine2coarse = {}
        for id in unsampled_classes:
            for coarse, fine in self.coarse2fine.items():
                if id in fine:
                    fine2coarse[id] = coarse
        assert len(fine2coarse) == len(unsampled_classes)

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
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, 1))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        query_xs = query_xs.reshape((-1, 1))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        query_xs = torch.stack(list(map(lambda x: self.test_transform(self.loader(x.squeeze(0)[0])), query_xs)))

        self.query_xs = query_xs
        self.query_ys = query_ys

    def __getitem__(self, item):
        query_xs = self.query_xs[item]
        query_ys = self.query_ys[item]
        return query_xs, query_ys

    def __len__(self):
        return len(self.query_ys)