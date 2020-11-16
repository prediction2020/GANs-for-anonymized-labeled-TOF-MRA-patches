import torch
import torch.nn as nn
import torchgan.layers as tgl

import config as c


# custom weights initialization called on netG and netD

def weights_init(m):
    
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('SpectralNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        
        
# Generator Code

class Generator(nn.Module):
  
    def __init__(self):
        super(Generator, self).__init__()
        if c.spectral_norm_G:
            self.convtrans1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.utils.spectral_norm(nn.ConvTranspose2d(c.nz, c.ngf * 16, c.kg, 1, 1, bias=False)),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf*16) x 3 x 3
            self.convtrans2 = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(c.ngf * 16, c.ngf * 8, c.kg, 2, 2, 1, bias=False)),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf*8) x 6 x 6
            self.convtrans3 = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(c.ngf * 8, c.ngf * 4, c.kg, 2, 2, 1, bias=False)),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf*4) x 12 x 12
            self.convtrans4 = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(c.ngf * 4, c.ngf * 2, c.kg, 2, 2, 1, bias=False)),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf) x 24 x 24
            self.convtrans5 = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(c.ngf * 2, c.ngf, c.kg, 2, 2, 1, bias=False)),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf) x 48 x 48
            self.convtrans6 = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(c.ngf, c.nc, c.kg, 2, 2, 1, bias=False)))
            self.activationG = nn.Tanh()
        else:
            self.convtrans1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(c.nz, c.ngf * 16, c.kg, 1, 1, bias=False),
                nn.BatchNorm2d(c.ngf * 16),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf*16) x 3 x 3
            self.convtrans2 = nn.Sequential(
                nn.ConvTranspose2d(c.ngf * 16, c.ngf * 8, c.kg, 2, 2, 1, bias=False),
                nn.BatchNorm2d(c.ngf * 8),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf*8) x 6 x 6
            self.convtrans3 = nn.Sequential(
                nn.ConvTranspose2d(c.ngf * 8, c.ngf * 4, c.kg, 2, 2, 1, bias=False),
                nn.BatchNorm2d(c.ngf * 4),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf*4) x 12 x 12
            self.convtrans4 = nn.Sequential(
                nn.ConvTranspose2d(c.ngf * 4, c.ngf * 2, c.kg, 2, 2, 1, bias=False),
                nn.BatchNorm2d(c.ngf * 2),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf) x 24 x 24
            self.convtrans5 = nn.Sequential(
                nn.ConvTranspose2d(c.ngf * 2, c.ngf, c.kg, 2, 2, 1, bias=False),
                nn.BatchNorm2d(c.ngf),
                nn.ReLU(True))
                #nn.LeakyReLU(0.2, inplace=True))
            # state size. (ngf) x 48 x 48
            self.convtrans6 = nn.Sequential(
                nn.ConvTranspose2d(c.ngf, c.nc, c.kg, 2, 2, 1, bias=False))
            self.activationG = nn.Tanh()
        
        # state size. (nc) x 96 x 96)
        
            
    def forward(self, inp):
        
        x = self.convtrans1(inp)
        x = self.convtrans2(x)
        x = self.convtrans3(x)
        x = self.convtrans4(x)
        x = self.convtrans5(x)
        last_conv_out = self.convtrans6(x)
        tanh_out = self.activationG(last_conv_out)
        return tanh_out


# Discriminator Code
# kernel_size = 5

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()

        if c.spectral_norm_D:
            self.conv1 = nn.Sequential(
                # input is (nc) x 96 x 96
                nn.utils.spectral_norm(nn.Conv2d(c.nc, c.ndf, c.kd, 2, 2, bias=False)),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf) x 48 x 48
            self.conv2 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(c.ndf, c.ndf * 2, c.kd, 2, 2, bias=False)),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*2) x 24 x 24
            self.conv3 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(c.ndf * 2, c.ndf * 4, c.kd, 2, 2, bias=False)),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*4) x 12 x 12
            self.conv4 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(c.ndf * 4, c.ndf * 8, c.kd, 2, 2, bias=False)),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*8) x 6 x 6
            # self.conv5 = nn.Sequential(
            #    nn.Conv2d(c.ndf * 8, c.ndf * 16, c.kd, 2, 2, bias=False),
            #    nn.BatchNorm2d(c.ndf * 16),
            #    nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*16) x 3 x 3
            # self.conv6 = nn.Conv2d(c.ndf * 16, 1, c.kd, 2, 1, bias=False)
            if c.mbdiscr:      
                self.conv5 = nn.Sequential(
                    nn.utils.spectral_norm(nn.Conv2d(c.ndf * 8, 50, c.kd, 2, 1, bias=False)))
                # self.conv6 = nn.Conv2d(c.ndf * 16, 1, c.kd, 2, 1, bias=False)
                self.lin1 = nn.Linear(200, 200)  # nn.Linear(200, 100)
                self.mbd = tgl.MinibatchDiscrimination1d(200, 100)  # tgl.MinibatchDiscrimination1d(100,50)
                self.lin2 = nn.Linear(300, 1)  # nn.Linear(150, 1)
            else:
                self.conv5 = nn.Sequential(
                    nn.utils.spectral_norm(nn.Conv2d(c.ndf * 8, c.ndf * 16, c.kd, 2, 2, bias=False)),
                    nn.LeakyReLU(0.2, inplace=True))
                self.conv6 = nn.Conv2d(c.ndf * 16, 1, c.kd, 2, 1, bias=False)
        else:
            self.conv1 = nn.Sequential(
                # input is (nc) x 96 x 96
                nn.Conv2d(c.nc, c.ndf, c.kd, 2, 2, bias=False),
                nn.InstanceNorm2d(c. ndf),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf) x 48 x 48
            self.conv2 = nn.Sequential(
                nn.Conv2d(c.ndf, c.ndf * 2, c.kd, 2, 2, bias=False),
                nn.InstanceNorm2d(c.ndf * 2),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*2) x 24 x 24
            self.conv3 = nn.Sequential(
                nn.Conv2d(c.ndf * 2, c.ndf * 4, c.kd, 2, 2, bias=False),
                nn.InstanceNorm2d(c.ndf * 4),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*4) x 12 x 12
            self.conv4 = nn.Sequential(
                nn.Conv2d(c.ndf * 4, c.ndf * 8, c.kd, 2, 2, bias=False),
                nn.InstanceNorm2d(c.ndf * 8),
                nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*8) x 6 x 6
            # self.conv5 = nn.Sequential(
            #    nn.Conv2d(c.ndf * 8, c.ndf * 16, c.kd, 2, 2, bias=False),
            #    nn.BatchNorm2d(c.ndf * 16),
            #    nn.LeakyReLU(0.2, inplace=True))
            # state size. (ndf*16) x 3 x 3
            # self.conv6 = nn.Conv2d(c.ndf * 16, 1, c.kd, 2, 1, bias=False)
            if c.mbdiscr:      
                self.conv5 = nn.Sequential(
                    nn.Conv2d(c.ndf * 8, 50, c.kd, 2, 1, bias=False))
                # self.conv6 = nn.Conv2d(c.ndf * 16, 1, c.kd, 2, 1, bias=False)
                self.lin1 = nn.Linear(200, 200)  # nn.Linear(200, 100)
                self.mbd = tgl.MinibatchDiscrimination1d(200, 100)  # tgl.MinibatchDiscrimination1d(100,50)
                self.lin2 = nn.Linear(300, 1)  # nn.Linear(150, 1)
            else:
                self.conv5 = nn.Sequential(
                    nn.Conv2d(c.ndf * 8, c.ndf * 16, c.kd, 2, 2, bias=False),
                    nn.InstanceNorm2d(c.ndf * 16),
                    nn.LeakyReLU(0.2, inplace=True))
                self.conv6 = nn.Conv2d(c.ndf * 16, 1, c.kd, 2, 1, bias=False)
    

    def forward(self, inp):
        
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if c.mbdiscr:
            last_conv_output = self.conv5(x)
            x = last_conv_output.view(-1, self.num_flat_features(last_conv_output))
            x = self.lin1(x)
            mbd_layer = self.mbd(x)
            sig_out = self.lin2(mbd_layer)
        else:
            x = self.conv5(x)
            sig_out = self.conv6(x)
        return sig_out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

