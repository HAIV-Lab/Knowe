# Coarse-To-Fine Incremental Few-Shot Learning

<a href="https://arxiv.org/abs/2111.14806">![](https://img.shields.io/badge/arxiv-2111.14806-red.svg)</a>

> Different from fine-tuning models pre-trained on a large-scale dataset of preset classes, class-incremental learning (CIL) aims to recognize novel classes over time without forgetting pre-trained classes. However, a given model will be challenged by test images with finer-grained classes, e.g., a basenji is at most recognized as a dog. Such images form a new training set (i.e., support set) so that the incremental model is hoped to recognize a basenji (i.e., query) as a basenji next time. This paper formulates such a hybrid natural problem of coarse-to-fine few-shot (C2FS) recognition as a CIL problem named C2FSCIL, and proposes a simple, effective, and theoretically-sound strategy Knowe: to learn, freeze, and normalize a classifier's weights from fine labels, once learning an embedding space contrastively from coarse labels. Besides, as CIL aims at a stability-plasticity balance, new overall performance metrics are proposed. In that sense, on CIFAR-100, BREEDS, and tieredImageNet, Knowe outperforms all recent relevant CIL or FSCIL methods.

### Description

Official PyTorch implementation of "Coarse-To-Fine Incremental Few-Shot Learning"

### Installation

- Clone this repo:
  
  ```
  git clone https://github.com/HAIV-Lab/Knowe.git
  cd Knowe
  ```
  
- Install required packages:
  
  ```
  pip install -r requirements.txt
  ```
  

### Data Preparation

#### BREEDS

1. Download the **ImageNet** dataset
  
2. Download BREEDS file from [official BREEDS repo](https://github.com/MadryLab/BREEDS-Benchmarks/blob/master/Constructing%20BREEDS%20datasets.ipynb)
  
3. Final folder structure should be:
  
  ```
  └── breeds_root
   ├── BREEDS
   │   ├── class_hierarchy.txt
   │   ├── dataset_class_info.json
   │   └── node_names.txt
   └── Data
       └── CLS-LOC
              ├── train
              │    └── n15075141
              │            .
              │            .
              │            .
              └── val
                   └── n15075141
                           .
                           .
                           .
  ```
  
  #### tieredImageNet
  
  [[Google Drive](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07)]
  

- Taken from [Meta-Learning for Semi-Supervised Few-Shot Classification](https://github.com/renmengye/few-shot-ssl-public)
  
- Final folder structure should be:
  
  ```
  tieredImageNet_root
      └── tieredImageNet
          ├── train_labels.pkl
          ├── val_labels.pkl
          ├── test_labels.pkl
          ├── train_images_png.pkl
          ├── val_images_png.pkl
          ├── test_images_png.pkl
          ├── synsets.txt
          └── class_names.txt
  ```
  

#### CIFAR-100

[[Direct Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)]

- Taken from [[the official webpage](https://www.cs.toronto.edu/~kriz/cifar.html)].
  
- Final folder structure should be:
  
  ```
  └── cifar_root
      └── cifar-100-python
          ├── meta
          ├── train
          └── test
  ```
  

### Pretrained Models

All these pretrained models are used as the parameter of `model_dir`

| dataset | Contrastive | Normalization | download |
| --- | --- | --- | --- |
| LIVING-17 | False | True | [link](https://drive.google.com/file/d/1dkHcSEJ0fN9XZkWHqXlDySAXlQchU3SR/view?usp=sharing) |
| LIVING-17 | True | False | [link](https://drive.google.com/file/d/1mN2CcUoKj7vHh3vHjCjfar-9gHvHCgvt/view?usp=sharing) |
| LIVING-17 | True | True | [link](https://drive.google.com/file/d/1DwJYAMAHTZ_RVtn7LZ6COKw9siDt1VM2/view?usp=sharing) |
| NONLIVING-26 | True | False | [link](https://drive.google.com/file/d/11LEg3HaBAqiDLH6hWhFVr4A9ax_vTnYn/view?usp=sharing) |
| NONLIVING-26 | True | True | [link](https://drive.google.com/file/d/1yvS07Qu4PSmM9jbsuyqaDQBC5Shk0CZm/view?usp=sharing) |
| ENTITY-13 | True | False | [link](https://drive.google.com/file/d/1UBsOWAkXp8FZdwOhWezWiHVbBF5uMxge/view?usp=sharing) |
| ENTITY-13 | True | True | [link](https://drive.google.com/file/d/1fskNXokS89SAk_47Ck5MTqRLhsJtbOBa/view?usp=sharing) |
| ENTITY-30 | True | False | [link](https://drive.google.com/file/d/1uzsgdJGjviYA24HpNy4V93TVSw0AfWi-/view?usp=sharing) |
| ENTITY-30 | True | True | [link](https://drive.google.com/file/d/12NtBZ_r1iGMf84UdA0mpD17i_pBnXkiB/view?usp=sharing) |
| tieredImageNet | True | False | [link](https://drive.google.com/file/d/17LzhzUzmw2GoPwBHSISBuJIK-ahruh80/view?usp=sharing) |
| tieredImageNet | True | True | [link](https://drive.google.com/file/d/1MJB8UQZbuA1oJQd7waG_fnSoYEJ1iS3A/view?usp=sharing) |
| CIFAR100 | True | False | [link](https://drive.google.com/file/d/1zmBJBP3P24v9dRjM41TRA8r-SoAUeEGg/view?usp=sharing) |
| CIFAR100 | True | True | [link](https://drive.google.com/file/d/1iNQPzwHClE3XCCUhC19rfe40whWohMf0/view?usp=sharing) |

### Training using our pretrained model

When running, only need `cifar_root` or `tieredImageNet_root` or `breeds_root`.

The Method can choose from `no MoCo`, `FT weight`, `no Norm`, `FT FC`, `ANCOR`, `ScaIL`, `LwF`, `subspace`, `align`, `Knowe` and `upperbound`.

```
python main.py \
-dataset cifar100 \
-cifar_root [path/to/cifar100] \
-tieredImageNet_root [path/to/tieredImageNet] \
-breeds_root [path/to/BREEDS] \
-method [choose one] \
-epochs_base 1 \
-model_dir [model we provided]
```

**NOTES**:

- `model_dir` must be the same True or False of Contrastive and Normalization with methods in our paper
- `ScaIL` can only run after `ANCOR`
- only living17 can choose `no MoCo`, `FT weight`, `no Norm`, `FT FC`

### Training from scratch

First, use [official ANCOR repo](https://github.com/guybuk/ANCOR) to train a Contrastive model

```
python train.py \
--world-size 1  --rank 0 -p 1 --cos --mlp \
--dataset cifar100 --mode coarse --data [path/to/cifar] \
--gpu 0 --batch-size 128 --multiprocessing-distributed \
--arch resnet12forcifar
```

Then, run our code using

```
python main.py \
-dataset cifar100 \
-cifar_root [path/to/cifar100] \
-tieredImageNet_root [path/to/tieredImageNet] \
-breeds_root [path/to/BREEDS] \
-method [choose one] \
-pretrained [ANCOR model]
```

### References

[1] Xiang Xiang, Yuwen Tan, Qian Wan, Jing Ma, Alan L. Yuille, Gregory D. Hager: Coarse-To-Fine Incremental Few-Shot Learning. To appear at ECCV 2022. https://arxiv.org/abs/2111.14806
