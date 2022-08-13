# PyTorch VQGAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Shubhamai/pytorch-vqgan/blob/main/LICENSE)
[![Run Python Tests](https://github.com/Shubhamai/pytorch-vqgan/actions/workflows/main.yml/badge.svg)](https://github.com/Shubhamai/pytorch-vqgan/actions/workflows/main.yml)

<p align="center">
<img src="./utils/assets/vqgan.png"/><br>
<em>Figure 1. VQGAN Architecture</em>
</p>

> **Note:** This is a work in progress.


This repo contains the implementation of the VQGAN - *[Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2010.11929)* in PyTorch from scratch. I wanted to create this repo to better understand VQGAN myself, but to also to provide scripts for faster training and experimentation with a toy dataset like MNIST etc. I also tried to make it as clean as possible, with comments, logging, and testing, custom dataloaders & visualizations, etc.   

- [PyTorch VQGAN](#pytorch-vqgan)
  - [What is VQGAN?](#what-is-vqgan)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Training](#training)
    - [Testing](#testing)
    - [Visualizing](#visualizing)
  - [Notes](#notes)
    - [TODOs](#todos)
  - [Hardware requirements](#hardware-requirements)
  - [Shoutouts](#shoutouts)
  - [BibTeX](#bibtex)

## What is VQGAN?



## Setup 


## Usage

### Training

### Testing

### Visualizing

## Notes

### TODOs  

## Hardware requirements

## Shoutouts

The list here contains some helpful blogs or videos that helped me a bunch in understanding the VQGAN.

1. [The Illustrated VQGAN](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) by Lj Miranda
2. [VQGAN: Taming Transformers for High-Resolution Image Synthesis [Paper Explained]](https://youtu.be/-wDSDtIAyWQ) by Gradient Dude
3. [VQ-GAN: Taming Transformers for High-Resolution Image Synthesis | Paper Explained](https://youtu.be/j2PXES-liuc) by The AI Epiphany
4. [VQ-GAN | Paper Explanation](https://youtu.be/wcqLFDXaDO8) and [VQ-GAN | PyTorch Implementation](https://youtu.be/_Br5WRwUz_U) by Outlier


## BibTeX

```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
