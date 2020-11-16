
# ####### Train settings ####### #

# Root directory for dataset

trial_num = 3

nr_patients = 41

n_disc = 5
lambdaa = 10


is_model_saved = True
patch_type = 'random'

DL = 4
if DL == 3:
    dataroot = "/home/tabea/Documents/Data/"

    save_model = "/home/tabea/Documents/Pooja/WGAN_GP/Models_random_patches/Trial_"+str(trial_num)+"/"
    save_results = "/home/tabea/Documents/Pooja/WGAN_GP/Results_random_patches/Trial_"+str(trial_num)+"/"

else:
    # path to data folder
    dataroot = "/fast/users/kossent_c/work/tabea/data/"

    # Paths for saving results and models
    save_model = "/fast/users/kossent_c/work/tabea/models/WGAN_" + str(trial_num) + "/"
    save_results = "/fast/users/kossent_c/work/tabea/results/WGAN_" + str(trial_num) + "/"

# Number of workers for dataloader
workers = 2

# Number of images
num_images = nr_patients*1000

# Batch size during training
batch_size = 300 #500  

# kernel sizes
kd = 5
kg = 5


# Spatial size of training images. All images need to be of the same size
image_size = 96

# Number of channels in the training images. 2 here as we are also using the labels through the second channel
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 128  

# Size of feature maps in generator
ngf = 96

# Size of feature maps in discriminator
ndf = 96

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lrd = 0.0001  # default: 0.0004
lrg = 0.0001  # default: 0.0004

# Minibatch discrimination on/off
mbdiscr = False

# Beta1 hyperparameter for Adam optimizers
beta1d = 0
beta1g = 0
beta2d = 0.9
beta2g = 0.9

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# gpu number
cuda_n = [0]

# #### Stability parameters #### #

# noise_type = 'uniform'
noise_type = 'gaussian'

# instance noise
instance_noise = False
sigma = 1
lin_anneal = sigma/(num_epochs*num_images/batch_size)

# spectral normalization
spectral_norm_D = False
spectral_norm_G = False


# ###### Evaluation related ###### #
# trial number and epoch to be loaded to generate test images

load_trial = 2
load_epoch = 180

gen_threshold = 0.8  # threshold for the generated labels
n_test_samples = 41000

if patch_type == 'random':

    saved_model_path = "/fast/users/kossent_c/work/tabea/models/WGAN_" + \
                       str(load_trial) + "/" + "epoch_" + str(load_epoch) + \
                       ".pth"

    save_test_path = "/fast/users/kossent_c/work/tabea/results/test_images/Test_" + \
                     str(gen_threshold) + "_" + \
                     str(load_trial) + "_" + str(load_epoch) + "/"
else:
    saved_model_path = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Models/Trial_"+str(load_trial)+"/" \
                      + "epoch_"+str(load_epoch)+".pth"
    save_test_path = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Results/test_images/Test_" \
                     + str(gen_threshold)+"_" \
                    + str(load_trial)+"_"+str(load_epoch) + "/"
