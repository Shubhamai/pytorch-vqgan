# PyTorch VQGAN

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Shubhamai/pytorch-vqgan/blob/main/LICENSE)
[![Run Python Tests](https://github.com/Shubhamai/pytorch-vqgan/actions/workflows/main.yml/badge.svg)](https://github.com/Shubhamai/pytorch-vqgan/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/Shubhamai/pytorch-vqgan/branch/main/graph/badge.svg?token=NANKT1FU4M)](https://codecov.io/gh/Shubhamai/pytorch-vqgan)

<p align="center">
<img src="./utils/assets/vqgan.png"/><br>
<em>Figure 1. VQGAN Architecture</em>
</p>

> **Note:** This is a work in progress.


This repo purpose is to serve as a more cleaner and feature-rich implementation of the VQGAN - *[Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2010.11929)* from the initial work of [dome272's repo](https://github.com/dome272/VQGAN-pytorch) in PyTorch from scratch. There's also a great video on the [explanation of VQGAN](https://youtu.be/wcqLFDXaDO8) by dome272.  

I created this repo to better understand VQGAN myself, and to provide scripts for faster training and experimentation with a toy dataset like MNIST etc. I also tried to make it as clean as possible, with comments, logging, testing & coverage, custom datasets & visualizations, etc.   

- [PyTorch VQGAN](#pytorch-vqgan)
  - [What is VQGAN?](#what-is-vqgan)
    - [Stage 1](#stage-1)
    - [Training](#training)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Training](#training-1)
    - [Testing](#testing)
    - [Visualizing](#visualizing)
    - [Tests](#tests)
  - [Notes](#notes)
    - [TODOs](#todos)
  - [Hardware requirements](#hardware-requirements)
  - [Shoutouts](#shoutouts)
  - [BibTeX](#bibtex)

## What is VQGAN?

VQGAN stands for **V**ector **Q**uantised **G**enerative **A**dversarial **N**etworks. The main idea behind this paper is to use CNN to learn the visual part of the image and generate a codebook of context-rich visual parts and then use Transformers to learn the long-range/global interactions between the visual parts of the image embedded in the codebook. Combining these two, we can generate very high-resolution images.

Learning both of these short and long-term interactions to generate high-resolution images is done in two different stages. 

1. The first stage uses VQGAN to learn the codebook of **context-rich** visual representation of the images. In terms of architecture, it is very similar to VQVAE in that it consists of an encoder, decoder and the codebook. We will learn more about this in the next section. 
  <p align="center">
  <img src="./utils/assets/vqvae_arch.png" width="400"/><br>
  <em>Figure 2. VQVAE Architecture</em>
  </p>

2. Using a transformer to learn the global interactions between the vectors in the codebook, to generate high-resolution images. 



### Stage 1
<img align="right" src="./utils/assets/encoder_arch.png" width="350"/>
The architecture of VQGAN consists of majorly three parts, the encoder, decoder and the Codebook, similar to the VQVAE paper 


1. The encoder [`encoder.py`](vqgan/encoder.py) part in the VQGAN learns to represent the images into a much lower dimension called embeddings and consists of Convolution, Downsample, Residual blocks and special attention blocks ( Non-Local blocks ), around 30 million parameter in default settings. 
2. The embeddings are then quantized using CodeBook and the quantized embeddings are used as input to the decoder [`decoder.py`](vqgan/decoder.py) part. 
3. The decode takes those embeddings and reconstructs the images. The architecture is similar to the encoder but reversed. Around 40 million parameters in default settings, slightly more compared to encoder due to more number of residual blocks. 

### Training




## Setup 


## Usage

### Training

### Testing

### Visualizing

### Tests

I have also just started getting my feet wet with testing and automated testing with GitHub CI/CD, so the tests here might not be the best practices.

To run tests, run `pytest --cov-config=.coveragerc --cov=. test`

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
