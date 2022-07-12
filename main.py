import argparse
from util import *
from data import setting
from C2FSCILTrainer import C2FSCILTrainer

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('-dataset', type=str, default='cifar100',
                    choices=['tiered', 'living17', 'entity13', 'nonliving26', 'entity30', 'cifar100'])
parser.add_argument('-cifar_root', type=str, default=None)
parser.add_argument('-tieredImageNet_root', type=str, default=None)
parser.add_argument('-breeds_root', type=str, default=None)

# training
parser.add_argument('-epochs_base', type=int, default=200)
parser.add_argument('-epochs_new', type=int, default=200)
parser.add_argument('-lr_base', type=float, default=0.12)
parser.add_argument('-lr_new', type=float, default=0.1)
parser.add_argument('-lambd', type=int, default=0.5)
parser.add_argument('-batch_size_base', type=int, default=256)
parser.add_argument('-batch_size_new', type=int, default=256)
parser.add_argument('-test_batch_size', type=int, default=256)
parser.add_argument('-num_workers', type=int, default=8)

parser.add_argument('-method', type=str, default=None, required=True,
                    choices=['no MoCo', 'FT weight', 'no Norm', 'FT FC', 'ANCOR', 'ScaIL', 'LwF', 'subspace', 'align', 'Knowe', 'upperbound'])
parser.add_argument('-pretrained', type=str,
                    default=None,
                    help='loading model parameter from a pretrained ANCOR model (need to train fc)')
parser.add_argument('-model_dir', type=str,
                    default=None,
                    help='loading model parameter from a pretrained base model (fc pretrained)')

parser.add_argument('-gpu', default='0')
parser.add_argument('-seed', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    args = setting(args)
    trainer = C2FSCILTrainer(args)
    trainer.train()
