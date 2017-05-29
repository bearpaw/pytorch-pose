# PyTorch-Pose

PyTorch-Pose is a PyTorch implementation of the general pipeline for 2D single human pose estimation. The aim is to provide the interface of the training/inference/evaluation, and the dataloader with various data augmentation options for the most popular human pose databases (e.g., [the MPII human pose](http://human-pose.mpi-inf.mpg.de), [LSP](http://www.comp.leeds.ac.uk/mat4saj/lsp.html) and [FLIC](http://bensapp.github.io/flic-dataset.html)).

Some codes for data preparation and augmentation are brought from the [Stacked hourglass network](https://github.com/anewell/pose-hg-train). Thanks to the original author. 

## Features
- Multi-thread data loading
- Multi-GPU training
- Logger
- Training/testing results visualization

## Installation
Please follow the [installation instruction of PyTorch](http://pytorch.org/). Note that the code is developed with Python2 and has not been tested with Python3 yet. 

Create a symbolic link to the `images` directory of the MPII dataset:
```
ln -s PATH_TO_MPII_IMAGES_DIR data/mpii/images
```

## Usage
Run the following command in terminal to train a single stack of hourglass network on the MPII human pose dataset. 
```
CUDA_VISIBLE_DEVICES=0 python example/mpii.py -a hg1 -j 4 -b 12 --checkpoint checkpoint/mpii/hg1
```
Here, `CUDA_VISIBLE_DEVICES=0` identifies the GPU devices you want to use. For example, use `CUDA_VISIBLE_DEVICES=0,1` if you want to use two GPUs with ID `0` and `1`. `-j` specifies how many workers you want to use for data loading. `-b` specifies the size of the minibatch.

Please refer to the `example/mpii.py` for the supported options/arguments.


## To Do List
Supported dataset
- [x] MPII human pose
- [ ] FLIC
- [ ] LSP

Supported models
- [x] One-stack hourglass networks
- [ ] Stacked hourglass networks 

**I have trouble in implementing multiple stacks of hourglass networks (overfitting). Please create a pull request if you want to contribute.**





