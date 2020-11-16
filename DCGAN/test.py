import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.backends.cudnn as cudnn

import config as c
import model as md
import utils as ut


# Set random seed for reproducibility
manualSeed = 999
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True

# increase the speed of training if you are not varying the size of image after each epoch
torch.backends.cudnn.benchmark = True

# to fix error with matplotlib
plt.switch_backend('agg')

# to ensure it doesn't run partly on another gpu
torch.cuda.set_device(c.cuda_n[0])


# Device selection
device = torch.device("cuda:"+str(c.cuda_n[0]) if(torch.cuda.is_available() and
                                                  c.ngpu > 0) else "cpu")


# ####Create generator object##### #
netG = md.Generator().to(device)

# Setup same optimizers and parameters as the trial run being evaluated for G
optimizerG = optim.Adam(netG.parameters(), lr=c.lrg, betas=(c.beta1g, 0.999))

# Print the model
print(netG)

saved_params_dict = torch.load(c.saved_model_path)

netG.load_state_dict(saved_params_dict['Generator_state_dict'])
optimizerG.load_state_dict(saved_params_dict['OptimizerG_state_dict'])

# number of noise images to
if c.noise_type == 'uniform':
    test_noise = torch.empty(c.n_test_samples, c.nz, 1, 1).uniform_(-1, 1)
else:
    test_noise = torch.randn(c.n_test_samples, c.nz, 1, 1)

dataloader = data_utils.DataLoader(test_noise, batch_size=1024,
                                   shuffle=False)

test_fake = torch.empty(c.n_test_samples, 2, c.image_size, c.image_size)
for i, data in enumerate(dataloader):
    noise = data.to(device)
    with torch.no_grad():
        if i != len(dataloader)-1:
            test_fake[i*1024:(i+1)*1024] = netG(noise).detach().cpu()
        else:
            test_fake[i*1024:c.n_test_samples] = netG(noise).detach().cpu()   
  

# save generated images as jpeg

for i, fake in enumerate(test_fake):
    #plt.figure()
    # hard thresholding for visualisation
    sample = fake.clone()
    sample[1][sample[1] > c.gen_threshold] = 1
    sample[1][sample[1] <= c.gen_threshold] = 0
    sample[0] = ut.rescale_unet(sample[0])  # rescaling back to 0-255
    test_fake[i] = sample
    # print out png images only when few images to be visualised.
    #plt.imshow(torch.cat((sample[0], fake[1], sample[1]), dim=1),
    #          cmap='gray', vmin=-1, vmax=1, animated=True)
    #plt.axis("off")
    #plt.savefig(c.save_test_path + "test_sample_%d_epoch_%d_trial_%d.png" % (i+1, c.load_epoch, c.load_trial))
    #plt.close()

# save all generated images as npy compression
gan_img = test_fake[:, 0, :, :].cpu().numpy()
gan_label = test_fake[:, 1, :, :].cpu().numpy()
np.savez_compressed(c.save_test_path + "test_epoch_%d_trial_%d_%d" % (c.load_epoch, c.load_trial, c.n_test_samples),
                    img=gan_img[:, :, :, np.newaxis], label=gan_label[:, :, :, np.newaxis])

img = gan_img[:, :, :, np.newaxis] 
label = gan_label[:, :, :, np.newaxis]
nr_imgs = 100
for i in range(nr_imgs):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img[i,:,:,0],cmap="gray")
    plt.title("Generated image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(label[i,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Label")
    plt.savefig(c.save_test_path + str(i))
    plt.close()
