# FastDVDnet

A state-of-the-art, simple and fast network for Deep Video Denoising which uses no motion compensation.

Oral presentation at CVPR2020. [CVPR publication page](https://openaccess.thecvf.com/content_CVPR_2020/html/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.html)

Previous deep video denoising algorithm: [DVDnet](https://github.com/m-tassano/dvdnet)

## Overview

This source code provides a PyTorch implementation of the FastDVDnet video denoising algorithm, as in 
Tassano, Matias and Delon, Julie and Veit, Thomas. ["FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation", arXiv preprint arXiv:1907.01361 (2019).](https://arxiv.org/abs/1907.01361)

## Datasets and Video Examples

You can download several denoised sequences with our algorithm and other methods [here](https://www.dropbox.com/sh/m9mpz1m1b55x420/AAAt1wes43brv37BmBxw07jna?dl=0 "FastDVDnet denoised sequences")

### Trainset

The 2017 DAVIS dataset was used for training.
You can find a list with the names of the 480p sequences employed [here](https://gist.github.com/m-tassano/27c1ef00ca42a8e50c2cee8a4205e559).
The dataloader needs the sequences in mp4 format. You can find the converted .mp4 files under the _training_ folder [here](https://www.dropbox.com/sh/m9mpz1m1b55x420/AAAt1wes43brv37BmBxw07jna?dl=0 "FastDVDnet denoised sequences").

Note: when converting the sequences one has to pay particular attention to the 'crf' and 'keyint' ffmpeg parameters to avoid strong compression. For the code to convert the image sequences see this [gist](https://gist.github.com/m-tassano/0536391eb79d63864e5005ea4da88243)

### Testsets

Two testsets are used in the paper: _Set8_ and the _2017 DAVIS testset_.

_Set8_ is composed of 8 sequences: 4 sequences from the _Derf 480p_ testset ("tractor", "touchdown", "park_joy", "sunflower") plus other 4 540p sequences. You can find these under the _test_sequences_ folder [here](https://www.dropbox.com/sh/m9mpz1m1b55x420/AAAt1wes43brv37BmBxw07jna?dl=0 "FastDVDnet denoised sequences").


## Running Times

FastDVDnet is orders of magnitude faster than other state-of-the-art methods

<img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/runtimes_all_log.png" width=350>

## Results

<img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/9831-teaser.gif" width=256> <img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/psnrs.png" width=600>

Left: Input noise sigma 40 denoised with FastDVDnet (sorry about the dithering due to gif compression)

Right: PSNRs on the DAVIS testset, Gaussian noise and clipped Gaussian noise

## Architecture

<img src="https://github.com/m-tassano/fastdvdnet/raw/master/img/arch.png" heigth=350>

## Code User Guide

### Colab example

You can use this [Colab notebook](https://colab.research.google.com/drive/1dPxlXPYgxanU-pgY4KOGsrCwSNo4IwBn?usp=sharing) to replicate the results

### Dependencies

The code runs on Python +3.6. You can create a conda environment with all the dependecies by running (Thanks to Antoine Monod for the .yml file)
```
conda env create -f requirements.yml -n <env_name>
```

Note: this project needs the [NVIDIA DALI](https://github.com/NVIDIA/DALI) package for training. The tested version of DALI is 0.10.0. If you prefer to install it yourself (supposing you have CUDA 10.0), you need to run
```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali==0.10.0 
```

### Testing

If you want to denoise an image sequence using the pretrained model you can execute

```
test_fastdvdnet.py \
	--test_path <path_to_input_sequence> \
	--noise_sigma 30 \
	--save_path results
```

**NOTES**
* The image sequence should be stored under <path_to_input_sequence>
* The model has been trained for values of noise in [5, 55]
* run with *--no_gpu* to run on CPU instead of GPU
* run with *--save_noisy* to save noisy frames
* set *max_num_fr_per_seq* to set the max number of frames to load per sequence
* to denoise _clipped AWGN_ run with *--model_file model_clipped_noise.pth*
* run with *--help* to see details on all input parameters

### Training

If you want to train your own models you can execute

```
train_fastdvdnet.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir logs
```

**NOTES**
* As the dataloader in based on the DALI library, the training sequences must be provided as mp4 files, all under <path_to_input_mp4s>
* The validation sequences must be stored as image sequences in individual folders under <path_to_val_sequences>
* run with *--help* to see details on all input parameters


## ABOUT

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Author    : Matias Tassano `mtassano at gopro dot com`
* Copyright : (C) 2019 Matias Tassano
* Licence   : GPL v3+, see GPLv3.txt

The sequences are Copyright GoPro 2018
