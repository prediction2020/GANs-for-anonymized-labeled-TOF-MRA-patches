import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd

import config as c


# #################################################### #
# ####### data pre-processing helper functions ####### #
# #################################################### #

def normalise(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def rescale_unet(x):   
    return 255 * (x - x.min()) / (x.max() - x.min())


# ################################################## #
# ######### Wasserstein helper function ########### #
# ################################################## #

def wasserstein_gradient_penalty(interpolate, d_interpolate, lambdaa):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(
        outputs=d_interpolate,
        inputs=interpolate,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradient_penalty = (gradients.norm(2) - 1) ** 2
    
    return gradient_penalty.mean() * lambdaa


# ################################################### #
# ####### Model architecture helper functions ####### #
# ################################################### #
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
