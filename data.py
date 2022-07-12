import os
import random

import numpy as np
import torch
import torchvision.transforms.transforms as transforms
from PIL import ImageFilter

from BREEDS import BREEDSFactory


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


train_tiered = [
            transforms.RandomResizedCrop(84, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4783, 0.4564, 0.4102],
                                 std=[0.2794756041544754, 0.27362885638931284, 0.28587171211901724, ])
        ]
meta_test_tiered = [
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4783, 0.4564, 0.4102],
                                 std=[0.2794756041544754, 0.27362885638931284, 0.28587171211901724, ])
        ]
test_tiered = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4783, 0.4564, 0.4102],
                                 std=[0.2794756041544754, 0.27362885638931284, 0.28587171211901724, ])
        ]
train_cifar = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]
test_cifar = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]
AUG = {
            "train_living17": [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "meta_test_living17": [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "test_living17": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "train_entity13": [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "meta_test_entity13": [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "test_entity13": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "train_nonliving26": [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "meta_test_nonliving26": [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "test_nonliving26": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])

            ],
            "train_entity30": [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "meta_test_entity30": [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
            "test_entity30": [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
            ],
        }


def setting(args):
    if args.method == 'no MoCo':
        args.contrastive = False
        args.decoupled = True
        args.norm = True
        args.memory = True
        args.upperbound = False
    elif args.method == 'FT weight':
        args.contrastive = True
        args.decoupled = False
        args.norm = True
        args.memory = True
        args.upperbound = False
    elif args.method == 'no Norm':
        args.contrastive = True
        args.decoupled = True
        args.norm = False
        args.memory = True
        args.upperbound = False
    elif args.method == 'FT FC':
        args.contrastive = True
        args.decoupled = True
        args.norm = True
        args.memory = False
        args.upperbound = False
    elif args.method in ['ANCOR', 'ScaIL', 'LwF']:
        args.contrastive = True
        args.decoupled = False
        args.norm = False
        args.memory = False
        args.upperbound = False
    elif args.method in ['align', 'subspace']:
        args.contrastive = True
        args.decoupled = True
        args.norm = False
        args.memory = True
        args.upperbound = False
    elif args.method in ['Knowe', 'upperbound']:
        args.contrastive = True
        args.decoupled = True
        args.norm = True
        args.memory = True
        args.upperbound = False if args.method == 'Knowe' else True
    if args.dataset == 'cifar100':
        import cifar as Dataset
        args.base_class = 20
        args.num_classes = 120
        args.fine_class = 100
        args.way = 10
        args.shot = 5
        args.sessions = 11
        args.query = 15
    elif args.dataset == 'tiered':
        import tieredImageNet as Dataset
        args.shot = 5
        args.query = 15
        args.base_class = 20
        args.way = 36
        args.sessions = 11
        args.num_classes = 371
        args.fine_class = 351
    elif args.dataset == 'living17':
        import BREEDS as Dataset
        args.base_class = 17
        args.num_classes = 85
        args.fine_class = 68
        args.way = 10
        args.shot = 1
        args.query = 15
        args.sessions = 8
    elif args.dataset == 'entity13':
        import BREEDS as Dataset
        args.base_class = 13
        args.num_classes = 273
        args.fine_class = 260
        args.way = 20
        args.shot = 1
        args.query = 15
        args.sessions = 14
    elif args.dataset == 'entity30':
        import BREEDS as Dataset
        args.base_class = 30
        args.num_classes = 270
        args.fine_class = 240
        args.way = 30
        args.shot = 1
        args.query = 15
        args.sessions = 9
    elif args.dataset == 'nonliving26':
        import BREEDS as Dataset
        args.base_class = 26
        args.num_classes = 130
        args.fine_class = 104
        args.way = 10
        args.shot = 1
        args.query = 15
        args.sessions = 12
    else:
        raise NotImplementedError
    args.Dataset = Dataset
    args.root_dir = f'log/{dir}_{args.dataset}_{args.way}way_{args.shot}shot_{args.query}query_{args.sessions}session/'
    args.saveweights = True if args.method != 'ScaIL' else False
    return args

def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.cifar_root, train=True, transform=transforms.Compose(train_cifar), download=True,
                                         index=class_index, base_sess=True, mode='coarse', args=args)
        testset = args.Dataset.CIFAR100(root=args.cifar_root, train=False, transform=transforms.Compose(test_cifar), download=False,
                                        index=class_index, base_sess=True, mode='coarse', args=args)
    elif args.dataset == 'tiered':
        trainset = args.Dataset.TieredImageNet(root=args.tieredImageNet_root, partition='train', mode='coarse',transform=transforms.Compose(train_tiered))
        testset = args.Dataset.TieredImageNet(root=args.tieredImageNet_root, partition='train', mode='coarse',transform=transforms.Compose(test_tiered))
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        breeds_factory = BREEDSFactory(info_dir=os.path.join(args.breeds_root, 'BREEDS'),
                                       data_dir=os.path.join(args.brreds_root, 'Data', 'CLS-LOC'))
        trainset = breeds_factory.get_breeds(
            ds_name=args.dataset,
            partition='train',
            mode='coarse',
            transforms=transforms.Compose(AUG[f'train_{args.dataset}']),
            split=None
        )
        testset = breeds_factory.get_breeds(
            ds_name=args.dataset,
            partition='validation',
            mode='coarse',
            transforms=transforms.Compose(AUG[f'test_{args.dataset}']),
            split=None
        )

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    if args.dataset == 'cifar100':
        if session == 1:
            new_class = session * args.way
            class_index = np.arange(0, new_class)
        else:
            new_class_update = session * args.way
            new_class = (session - 1) * args.way
            class_index = np.arange(new_class, new_class_update)
            if args.upperbound is True:
                class_index = np.arange(0, new_class_update)
        class_new = np.arange(session * args.way)
        trainset = args.Dataset.CIFAR100(root=args.cifar_root, train=True, transform=transforms.Compose(train_cifar), download=False,
                                         index=class_index, base_sess=False, mode='fine', args=args)
        testset = args.Dataset.CIFAR100(root=args.cifar_root, train=False, transform=transforms.Compose(test_cifar), download=False,
                                        index=class_new, base_sess=False, mode='fine', args=args)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
    elif args.dataset == 'tiered':
        trainset = args.Dataset.TieredImageNetTrainDataset(
            args=args,
            partition='train',
            train_transform=transforms.Compose(meta_test_tiered),
            session=session
        )
        testset = args.Dataset.TieredImageNetTestDataset(
            args=args,
            partition='train',
            test_transform=transforms.Compose(test_tiered),
            session=session
        )
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
    elif args.dataset in ['living17', 'entity13', 'nonliving26', 'entity30']:
        breeds_factory = BREEDSFactory(info_dir=os.path.join(args.breeds_root, 'BREEDS'),
                                       data_dir=os.path.join(args.brreds_root, 'Data', 'CLS-LOC'))
        trainset = args.Dataset.BreedsSessionTrainDataset(
                args=args,
                dataset=breeds_factory.get_breeds(
                    ds_name=args.dataset,
                    partition='train',
                    mode='fine',
                    transforms=None,
                    split=None
                ),
                train_transform=transforms.Compose(AUG[f"meta_test_{args.dataset}"]),
                session=session
            )
        testset = args.Dataset.BreedsSessionTestDataset(
            args=args,
            dataset=breeds_factory.get_breeds(
                ds_name=args.dataset,
                partition='validation',
                mode='fine',
                transforms=None,
                split=None
            ),
            test_transform=transforms.Compose(AUG[f"test_{args.dataset}"]),
            session=session
        )
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError
    return trainset, trainloader, testloader
