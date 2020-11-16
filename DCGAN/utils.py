import numpy as np
import torch.nn as nn
import torch

import config as c


def normalise(x): 
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def rescale_unet(x):   
    return 255 * (x - x.min()) / (x.max() - x.min())


def label_flipping(y, prob_flip=c.prob_flipping):  
    n_flips = int(prob_flip*len(y))
    idx_chosen = np.random.randint(0, len(y), n_flips)
    y[idx_chosen] = 1-y[idx_chosen]
    return y
    

def generator_conv_params(in_channel=200, out_channel=96*16,
                          kernel_size=5, stride=2, padding=1, output_padding=0,
                          img_size=1):
    m = nn.ConvTranspose2d(in_channel, out_channel,
                           kernel_size, stride, padding, output_padding, bias=False)
    inp = torch.randn(1, in_channel, img_size)
    output = m(inp)
    return output.size()


def discriminator_conv_params(in_channel=2, out_channel=96,
                              kernel_size=5, stride=2, padding=1, output_padding=0,
                              img_size=96):
    m = nn.Conv2d(in_channel, out_channel,
                  kernel_size, stride, padding, output_padding, bias=False)
    inp = torch.randn(1, in_channel, img_size, img_size)
    output = m(inp)
    return output.size()
