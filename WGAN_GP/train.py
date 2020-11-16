import numpy as np
import matplotlib.pyplot as plt

import random
import time

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

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
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# increase the speed of training if you are not varying the size of image after each epoch
torch.backends.cudnn.benchmark = False

# to fix error with matplotlib
plt.switch_backend('agg')

# to ensure it doesn't run partly on another gpu
torch.cuda.set_device(c.cuda_n[0])

# load numpy arrays with 41000 patches of size 96x96x1

if c.patch_type == 'random':
    data = np.load(c.dataroot + 'nr_patches_1000_random.npz')
    #data = np.load(c.dataroot + 'nr_patches_500_random1.npz')
else:
    data = np.load(c.dataroot + 'nr_patches_1000_most_err_patch_size_96_nr_pats_41_percent_fp_0_mode_fpfn_part_1.npz')


imgs = data['img']
mask = data['label']

if c.nr_patients<41:
    imgs = imgs[:c.nr_patients*1000]
    mask = mask[:c.nr_patients*1000]

# normalise the input images to range between [-1,1]
imgs_norm = np.array([ut.normalise(i) for i in imgs[:, :, :, 0]])

# convert the images and masks to tensors
tensor_imgs = torch.FloatTensor(imgs_norm)
tensor_mask = torch.FloatTensor(mask[:, :, :, 0])  # removing channel dimension for mask or label as well

# stack them together for the generator as 2 channels
train_pair = torch.stack((tensor_imgs, tensor_mask), 1)

dataset = data_utils.TensorDataset(train_pair)

dataloader = data_utils.DataLoader(dataset, batch_size=c.batch_size,
                                   shuffle=True, num_workers=c.workers)

# Device selection
device = torch.device("cuda:"+str(c.cuda_n[0]) if(torch.cuda.is_available() and
                                                  c.ngpu > 0) else "cpu")


# ####Create generator object##### #

netG = md.Generator().to(device)

# Handle multi-gpu if desired

if (device.type == 'cuda') and (c.ngpu > 1):
    netG = nn.DataParallel(netG, c.cuda_n)

# Apply the weights_init function to randomly initialize all weights

netG.apply(md.weights_init)

# Print the model
print(netG)


# #### Create discriminator object #### #

netD = md.Discriminator().to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (c.ngpu > 1):
    netD = nn.DataParallel(netD, c.cuda_n)

# Apply the weights_init function to randomly initialize all weights

netD.apply(md.weights_init)

# Print the model
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
if c.noise_type == 'uniform':
    fixed_noise = torch.empty(c.batch_size, c.nz, 1, 1, device=device).uniform_(-1, 1)
else:
    fixed_noise = torch.randn(c.batch_size, c.nz, 1, 1, device=device)


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=c.lrd, betas=(c.beta1d, c.beta2d))
optimizerG = optim.Adam(netG.parameters(), lr=c.lrg, betas=(c.beta1g, c.beta2g))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
Wasserstein_d = []
real_p = []
fake_mp_b4update = []
fake_mp_afterupdate = []
iters = 0
duration = 0
sigma = c.sigma

print("Starting Training Loop...")

# For each epoch
for epoch in range(c.num_epochs):
    epoch_start_time = time.time()
    # For each batch in the dataloader
    errD_iter = []
    errG_iter = []
    Wasserstein_d_iter = []
    D_x_iter = []
    D_G_z1_iter = []
    D_G_z2_iter = []
    
    for i, data in enumerate(dataloader, 0):
        errD_disc_iter = []
        Wasserstein_d_disc_iter = []
        D_x_disc_iter = []
        D_G_z1_disc_iter = []
        
        for _ in range(c.n_disc):
  
            # ########################## #
            # (1) Update D network
            # ######################### #
            # ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            
            # instance noise
            if c.instance_noise:
                ins_noise = torch.empty(real_cpu.size(), device=device).normal_(mean=0, std=sigma)
                errD_real = netD(real_cpu + ins_noise)
            else:
                # Forward pass real batch through D
                errD_real = netD(real_cpu)
                
            # Calculate loss on all-real batch
            errD_real = errD_real.view(-1).mean()*-1
            
            # Calculate gradients for D in backward pass
            errD_real.backward(retain_graph=True)
            
            # # Train with all-fake batch
            # Generate batch of latent vectors
            if c.noise_type == 'uniform':
                noise = torch.empty(b_size, c.nz, 1, 1, device=device).uniform_(-1, 1)  # uniform noise
            elif c.noise_type == 'gaussian':
                noise = torch.randn(b_size, c.nz, 1, 1, device=device)
            else:
                noise = torch.empty(b_size, c.nz, 1, 1, device=device)
                print("Please specify a valid distribution to sample noise vector from\n")
    
            # Generate fake image batch with G
    
            fake = netG(noise)
    
            if c.instance_noise:
                ins_noise = torch.empty(real_cpu.size(), device=device).normal_(mean=0, std=sigma)
                errD_fake = netD(fake.detach() + ins_noise)
            else:
                # Classify all fake batch with D
                errD_fake = netD(fake.detach())
    
            # Calculate D's loss on the all-fake batch
            errD_fake = errD_fake.view(-1).mean()
            
            # Calculate the gradients for this batch
            errD_fake.backward(retain_graph=True)
    
            # get epsilon value from uniform distribution for getting a sample from joint distribution
    
            eps = torch.rand(1).item()
    
            interpolate = eps * real_cpu + (1 - eps) * fake
            d_interpolate = netD(interpolate)
    
            # get gradient penalty
            gradient_penalty = ut.wasserstein_gradient_penalty(interpolate, d_interpolate, c.lambdaa)
    
            gradient_penalty.backward(retain_graph=True)
    
            # Update D
            optimizerD.step()
            errD = errD_fake + errD_real + gradient_penalty
            wasserstein = errD_fake + errD_real
            D_G_z1 = errD_fake.item()
            D_x = (errD_real * -1).item()
            
            # storing all the errors and outputs to be averaged over the discriminator iterations
            
            errD_disc_iter.append(errD.item())
            Wasserstein_d_disc_iter.append(wasserstein.item())
            D_G_z1_disc_iter.append(D_G_z1)
            D_x_disc_iter.append(D_x)
        
        # average all the errors and outputs of the whole n_disc iterations
        errD_disc_avg = np.mean(np.array(errD_disc_iter))
        Wasserstein_d_disc_avg = np.mean(np.array(Wasserstein_d_disc_iter))
        D_G_z1_disc_avg = np.mean(np.array(D_G_z1_disc_iter))
        D_x_disc_avg = np.mean(np.array(D_x_disc_iter))

        ############################
        # (2) Update G network
        ###########################
        
        netG.zero_grad()
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        if c.instance_noise:
            ins_noise = torch.empty(real_cpu.size(), device=device).normal_(mean=0, std=sigma)
            sigma = sigma - c.lin_anneal
            output_fake = netD(fake + ins_noise)
        else:
            output_fake = netD(fake)

        errG = -output_fake.view(-1).mean()
        
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output_fake.view(-1).mean().item()
        # Update G
        optimizerG.step()
        
        errD_iter.append(errD_disc_avg)
        errG_iter.append(errG.item())
        
        D_x_iter.append(D_x_disc_avg)
        D_G_z1_iter.append(D_G_z1_disc_avg)
        D_G_z2_iter.append(D_G_z2)
        Wasserstein_d_iter.append(Wasserstein_d_disc_avg)

        iters += 1
        if i % 100 == 0:
            print("[%d/%d] batches done!\n" % (i + 1, len(dataset) // c.batch_size))

    print(" End of Epoch %d \n" % epoch)

  
    # Output training stats averaged across batches after each epoch
    
    avg_errD = np.mean(np.array(errD_iter))
    avg_errG = np.mean(np.array(errG_iter))
    
    avg_D_x = np.mean(np.array(D_x_iter))
    avg_D_G_z1 = np.mean(np.array(D_G_z1_iter))
    avg_D_G_z2 = np.mean(np.array(D_G_z2_iter))
    
    avg_Wasserstein_d = np.mean(np.array(Wasserstein_d_iter))
    
    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
          % (epoch, c.num_epochs,
             avg_errD, avg_errG, avg_D_x, avg_D_G_z1, avg_D_G_z2))

    # Save Losses and outputs for plotting later
    G_losses.append(avg_errG.item())
    D_losses.append(avg_errD.item())

    real_p.append(avg_D_x)
    fake_mp_b4update.append(avg_D_G_z1)  # mean probability of classifying fake as real before updating D and G
    fake_mp_afterupdate.append(avg_D_G_z2)  # mean probability of classifying fake as real after updating D and G
    
    Wasserstein_d.append(avg_Wasserstein_d)

    np.save(c.save_results+'G_losses.npy', np.asarray(G_losses))
    np.save(c.save_results + 'D_losses.npy', np.asarray(D_losses))
    
    np.save(c.save_results+'real_p.npy', np.asarray(real_p))
    np.save(c.save_results+'fake_mp_b4update.npy', np.asarray(fake_mp_b4update))
    np.save(c.save_results+'fake_mp_afterupdate.npy', np.asarray(fake_mp_afterupdate))
    
    np.save(c.save_results+'Wasserstein_distance.npy', np.array(Wasserstein_d))

    # Check how the generator is doing by saving G's output on fixed_noise

    with torch.no_grad():
        fixed_fake = netG(fixed_noise).detach().cpu()
      
    img_list.append(fixed_fake.numpy())
    
    sample_idx = [10, 20, 30, 40]
    for idx in sample_idx:
        plt.figure()
        # hard thresholding for visualisation
        # sample = fixed_fake[idx].clone()
        # sample[1][sample[1] >= 0] = 1
        # sample[1][sample[1] < 0] = -1
        plt.imshow(torch.cat((fixed_fake[idx][0], fixed_fake[idx][1]), dim=1),
                   cmap='gray', vmin=-1, vmax=1, animated=True)
        plt.axis("off")
        plt.savefig(c.save_results+"fixed_fake_sample_%d_while_training_epoch_%d_.png" % (idx, epoch))
        plt.close()
      
    # save model parameters'
    if c.is_model_saved:
        
        torch.save({'Discriminator_state_dict': netD.state_dict(),
                    'Generator_state_dict': netG.state_dict(),
                    'OptimizerD_state_dict': optimizerD.state_dict(),
                    'OptimizerG_state_dict': optimizerG.state_dict()
                    }, c.save_model+"epoch_{}.pth".format(epoch))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(c.save_results+"losses.png")
    plt.close()
    
    epoch_end_time = time.time()
    
    duration = duration + (epoch_end_time-epoch_start_time)
    approx_time_to_finish = duration/(epoch+1)*(c.num_epochs-epoch)
    print("Training time for epoch ", epoch, ": ", (epoch_end_time-epoch_start_time)/60, " minutes.")
    print("Approximate time remaining for run to finish: ", approx_time_to_finish/3600, " hours")
