# ####### Train settings ####### #

trial_num = 57
nr_patients = 41
is_model_saved = True  # model is saved after epoch 60
patch_type = 'random'

# path to data folder
dataroot = "/home/tabea/Documents/Code/augmentation/error-based-GANs/Unet/train_data/"
# Paths for saving results and models
if patch_type == "random":
    save_model = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Models_random_patches/Trial_"+str(trial_num)+"/"
    save_results = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Results_random_patches/Trial_"+str(trial_num)+"/"
else:
    save_model = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Models/Trial_" + str(trial_num) + "/"
    save_results = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Results/Trial_" + str(trial_num) + "/"

# Number of workers for dataloader
workers = 2

# Number of images
num_images = nr_patients*1000

# Batch size during training
batch_size = 250 #500  

# kernel sizes
kd = 5
kg = 5

# Spatial size of training images. All images need to be of the same size
image_size = 96

# Number of channels in the training images. 2 here as we are also using the labels through the second channel
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 100  

# Size of feature maps in generator
ngf = 96

# Size of feature maps in discriminator
ndf = 96

# Number of training epochs
num_epochs = 300

# Learning rate for optimizers
lrd = 0.0003  # default: 0.0004
lrg = 0.0003  # default: 0.0004

# Minibatch discrimination on/off
mbdiscr = True

# Beta1 hyperparameter for Adam optimizers
beta1d = 0.5
beta1g = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# gpu number
cuda_n = [1]

# #### Stability parameters #### #

# noise_type = 'uniform'
noise_type = 'gaussian'

# label flipping choice
label_flipping_G = False
label_flipping_D = False
prob_flipping = 0.005

# label smoothing
label_smoothing = True

# feature matching
feature_matching = True
gen_criterion = "L1"
# gen_criterion = "MSE"

# instance noise
instance_noise = False
sigma = 1
lin_anneal = sigma/(num_epochs*num_images/batch_size)


# ###### Evaluation related ###### #
# trial number and epoch to be loaded to generate test images

load_trial = 50
load_epoch =  140
gen_threshold = 0.8  # threshold for the generated labels
n_test_samples = 41000

if patch_type == 'random':
    saved_model_path = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Models_random_patches/Trial_" + str(load_trial) + "/" \
                       + "epoch_" + str(load_epoch) + ".pth"
    save_test_path = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Results_random_patches/test_images/Test_" + \
                     str(gen_threshold) + "_" + \
                     str(load_trial) + "_" + str(load_epoch) + "/"
else:
    saved_model_path = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Models/Trial_"+str(load_trial)+"/" \
                       + "epoch_"+str(load_epoch)+".pth"
    save_test_path = "/home/tabea/Documents/Code/augmentation/error-based-GANs/DCGAN/Results/test_images/Test_" \
                     + str(gen_threshold)+"_" \
                     + str(load_trial)+"_"+str(load_epoch) + "/"
