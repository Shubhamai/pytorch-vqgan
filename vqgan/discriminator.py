"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)


This isn't a standward GAN discriminator, where the input is a batch of images and the output is a batch of real/fake labels.


But instead, PatchGAN discriminator is a network that takes a batch of images
split into multiple patches 

for ex. - 30-30 patches, 30 in x and 30 in y axis, similar to convolution kernels, 
and then runs them through a network to get a score of real/fake on those individual patches. 

ex. - input size (1, 3, 256, 256) -> output size (1, 1, 30, 30)

"""

import torch.nn as nn


class Discriminator(nn.Module):
    """  PatchGAN Discriminator


    Args:
        image_channels (int): Number of channels in the input image.
        num_filters_last (int): Number of filters in the last layer of the discriminator.
        n_layers (int): Number of layers in the discriminator.

    
    """
    
    def __init__(self, image_channels:int=3, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(image_channels, num_filters_last, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(
                    num_filters_last * num_filters_mult_last,
                    num_filters_last * num_filters_mult,
                    4,
                    2 if i < n_layers else 1,
                    1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True),
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)