
#################################################################################################################################################################################################################################################################################
def to_list_of_certain_size(input_number, number_of_elements):
    if type(input_number)==tuple or type(input_number)==list:
        if len(input_number)==1:
            return input_number*number_of_elements;
        else:
            return input_number;
    else:
        return [input_number]*number_of_elements


def make_list_of_certain_size(n_layers, *args):
    args_list = []
    for current_arg in args:
        if type(current_arg) != list:
            args_list.append([current_arg] * n_layers)
        else:
            assert len(current_arg) == n_layers, str(current_arg) + ' must have the same length as n_layers'
            args_list.append(current_arg)
    return args_list



class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self).__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

    def fit_line(self, y, n_jobs=1):
        len_input = y.shape[0]
        X = np.array(list(zip([1]*len_input,range(len_input))))
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        self.p_positive = stats.t.cdf(self.t, y.shape[0] - X.shape[1])
        return self









#### GPU Allocation Program: ####
default_device = device1
Generator_device = default_device
netF_device = default_device
discriminator_device = default_device

# Lambdas and stuff which don't participate in the loops and are the same for every set point:
lambda_DirectPixels_GradientSensitive = False;
lambda_GradientSensitive_low2high_ratio = 1;
flag_use_gradient_sensitive_loss = False


#####################################################################################################################################################################
### Get Original Networks and New Optimizer Optimizers: ###
flag_use_new_or_checkpoint_generator = 'new';  # 'new', 'checkpoint'
netF_name = 'resnet'  # 'vgg', 'resnet', 'resnet50dilated-ppm_deepsup'
number_of_conv_layers = 10
flag_use_fresh_discriminator_or_checkpoint_discriminator = 'fresh'
reference_image_load_paths_list = ['C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim01.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim02.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim03.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim04.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim05.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim15.png']
flag_initialize_hidden_with_zero_or_input = 'zero'; #'zero' / 'input'
flag_write_TB = True;
# flag_write_TB = False;


### Control Number of Training Iterations: ###
max_total_number_of_batches = 200000;  #10000
# max_total_number_of_batches = 50;  #10000
# flag_limit_number_of_iterations = True;
flag_limit_number_of_iterations = False;
noise_gain = 30;


### DEFINE SAVE filenames and folders: ###
# Generator:
setpoint_string = 'TEST1'
Generator_checkpoint_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl/TNR/Model New Checkpoints/';
# Generator_checkpoint_prefix = 'UNETV12_V3_MultiplicativeNoise' + 'alpha_estimator' + '_NoiseGain' + str(noise_gain);
Generator_checkpoint_prefix = 'UNETV14_V1_BayerNoise_LRScheduler_' + 'alpha_estimator' + '_NoiseGain' + str(noise_gain);
Generator_checkpoint_postfix = setpoint_string + '.pth';
Generator_checkpoint_folder = Generator_checkpoint_folder + Generator_checkpoint_prefix + '/'
# Discriminator:
Discriminator_checkpoint_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl/TNR/Model New Checkpoints/';
Discriminator_checkpoint_prefix = 'Discriminator_Network';
Discriminator_checkpoint_postfix = 'Gram_Loss_on_DirectPixels.pth';
Discriminator_checkpoint_folder = Discriminator_checkpoint_folder + Generator_checkpoint_prefix + '/'

### LOAD filenames and folders: ###
models_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Model New Checkpoints'
load_Discriminator_filename = 'Discriminator_Network_xintao.pth';
# load_Generator_filename = 'UNETV12_Mean2DExtra_V3_100_TEST1_Step1500_GenLR_8d00em05.pt';
load_Generator_filename = 'UNETV12_V2_BayerNoiseconvmean2d_running_average_NoiseGain30_TEST1_Step42200_GenLR_8d00em06.pt';

### Correct save name in case we're checkpoint mode: ###
if flag_use_new_or_checkpoint_generator == 'checkpoint':
    Generator_checkpoint_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl/TNR/Model New Checkpoints/';
    Generator_checkpoint_prefix = load_Generator_filename.split('_TEST')[0]
    Generator_checkpoint_postfix = setpoint_string + '.pth';
    Generator_checkpoint_folder = Generator_checkpoint_folder + Generator_checkpoint_prefix + '/'


######################
### Get Generator: ###
######################
lambda_learning_rate = 1;
learning_rate = 1e-1/5*lambda_learning_rate;
hypergrad_lr = 0*lambda_learning_rate;
if flag_use_new_or_checkpoint_generator == 'new':
    Generator_checkpoint_step = 0;
    Generator_Network = get_Original_Generator();  # number_of_conv_layers=-1 for all conv layers
    netF = get_netF(FeatureExtractor_name=netF_name, number_of_conv_layers_to_return=number_of_conv_layers)
    Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device)
    optimizer_G = get_Generator_Optimizer(Generator_Network, learning_rate, hypergrad_lr)
else:
    ### LOAD Network From Checkpoint If Wanted: ###
    Generator_checkpoint_step = int(load_Generator_filename.split('Step')[1].split('_')[0])   #Assumes only one Step substring in the Generator load filename
    Generator_Network = load_Generator_from_checkpoint(models_folder, load_Generator_filename, None);
    netF = get_netF(FeatureExtractor_name=netF_name, number_of_conv_layers_to_return=number_of_conv_layers)
    Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
    optimizer_G = get_Generator_Optimizer(Generator_Network, learning_rate, hypergrad_lr);

### Generator to train mode: ###
Generator_Network = Generator_to_train_mode(Generator_Network)
netF = freeze_gradients(netF)
# Generator_Network = freeze_non_temporal_layers(Generator_Network)
#####################################################################################################################################################################




#####################################################################################################################################################################
### Save Models/Images Frequency ([iterations]): ###
save_models_frequency = 100;
save_images_frequency = 30;

### Flags: ###
flag_train_or_not = True; ############################################################    VERY IMPORTANT NOW TO FORGET ABOUT THIS!!! this is 1 in the case i only want to show results or check something but without actualy training ######################
flag_residual_learning = False;
flag_clip_noisy_tensor_after_noising = False
flag_show_image_or_highpass_details = 1; # 0=image, 1=highpass details
flag_learn_clean_or_running_average = 0; # 0=clean, 1=running average


### Learning Rate Scheduling: ###
### Initialize loss value to keep track of best loss so far: ###
best_eval_loss = np.Inf
last_eval_loss = np.Inf
best_loss = np.Inf

### Initialize Learning Rate (later to be subject to lr scheduling): ###
flag_change_optimizer_betas = False
init_learning_rate = 2e-3 * 1
Generator_lr_previous = init_learning_rate
minimum_learning_rate = 1e-5
learning_rate_lowering_factor = 0.5;
learning_rate_lowering_patience_before_lowering = 100;
learning_rate_lowering_tolerance_factor_from_best_so_far = 0.1;
initial_spike_learning_rate_reduction_multiple = 0.75
minimum_learning_rate_counter = 0
learning_rate = init_learning_rate*(initial_spike_learning_rate_reduction_multiple**minimum_learning_rate_counter)  #TODO: make hyper-parameter  and instead of adjusting the initial LR at the start of each meta_epoch simply adjust it after each "meta-epoch" as i define it which would be dynamic
set_optimizer_learning_rate(optimizer_G, learning_rate)

### Fit a straight line to the loss value so far: ###
regressor = LinearRegression()
losses_straight_line_fit_check_frequency = 200;
losses_straight_line_fit_p_threshold = 0.999;
losses_straight_line_fit_LR_reduction_factor = 0.9


### Gradient Clipping: ###
#(1). Clip or Not:
# flag_use_gradient_clip_valueping = False;
flag_use_gradient_clip_valueping = True;
#(2). Clip Value or Norm:
# flag_clip_direct_grad_or_norm = 'direct'
flag_clip_direct_grad_or_norm = 'norm'
#(3). Clip Memory Cells or Network:
flag_clip_network_or_center = 'network'
lr_factor = 1;  #Old method of controlling initial learning rate...i'm setting this to 1
gradient_clip_value = 10; #Initialize with large value to avoid cutting off some unknown norm aprior... the EMA will take care of the rest
EMA_alpha = 0.05
ema_object = EMA(EMA_alpha)


### Decide Whether To Plot Gradient Flow through the network layers: ###
flag_plot_grad_flow = False


### Whether To Show Images Whilst Training: ###
flag_show_generated_images_while_Generator_training = False
# flag_show_generated_images_while_Generator_training = True
flag_show_generated_images_while_Discriminator_training = False;
#####################################################################################################################################################################







#############################################################################################################################
### Generator Loss Lambdas: ###
# (1). Global Lambda's Factor:
lambda_global_factor = 1 * flag_train_or_not;
#(2.1). Direct Pixels Loss Lambda:
lambda_DirectPixels = 1 * lambda_global_factor; #TODO: before it was 0.1
lambda_DirectPixels_Contextual = 1e-2 * lambda_global_factor;
lambda_DirectPixels_Gram = 5e5 * lambda_global_factor;
#(2.2). Gradient Sensitive Loss Lambda (Direct Pixels):
lambda_DirectPixels_GradientSensitive = 0 * lambda_global_factor
lambda_GradientSensitive_low2high_ratio = 0;
#(3). High Level (VGG/ResNet) Features Loss Lambda:
lambda_Features_Perceptual = 5e-1 * lambda_global_factor;
lambda_Features_Contextual = 0.5e-2 * lambda_global_factor; #TODO: before it was 3e-2
ContextualLoss_flag_L2_or_dotP = 2; #1=L2, 2=dotP
lambda_Features_Gram = 5e5 * lambda_global_factor;
# (4). Validity (Adversarial) Loss:
lambda_GAN_validity_loss = 0.001 * lambda_global_factor;
lambda_GradientPenalty_loss = 0.0001 * lambda_global_factor
# (5). Flags
#   (5.1). Generator:
flag_use_direct_pixels_L1_loss = True
flag_use_feature_extractor_direct_loss = False;
flag_use_feature_extractor_contextual_loss = False
flag_use_direct_pixels_contextual_loss = False;
flag_use_gradient_sensitive_loss = False
flag_use_direct_pixels_gram_loss = False;
flag_use_feature_extractor_gram_loss = False;
#   (5.2). Discriminator:
flag_use_validity_loss = False;  # Discriminator Loss!
flag_use_wgan_gp = False;  # Gradient Penalty Loss!
Discriminator_Loss_Type = 'wgan-gp'  # vanilla, lsgan, wgan-gp

### Flags and Thresholds: ###
# (1). High level flags:
flag_train_Generator = True
flag_train_Discriminator = False
flag_use_threshold_or_predetermined_GAN_policy = 'predetermined';  # 'threshold', 'predetermined'
# (2). Initial Steps:
number_of_initial_Generator_steps = 0
number_of_initial_Discriminator_steps = 0
# (3). Thresholds (loss/accuracy):
current_generator_loss = 0;
Generator_loss_threshold = -1;  # to activate we want current_generator_loss > Generator_loss_threshold
Discriminator_loss_threshold = -1;  # for BCE the min is 0, but for wgan the min if -inf....
Discriminator_accuracy_threshold = 100
current_discriminator_loss = 0  # to activate we want current_discriminator_loss > Discriminator_loss_threshold
current_discriminator_accuracy = 0
# (4). Predetermined:
discriminator_to_generator_training_steps = 1;  # TODO: swtich back to 5 or whatever else i decide

##########################
### Get Discriminator: ###
##########################
if flag_use_validity_loss:
    ### Get "Fresh" Discriminator: ###
    Discriminator_Network = Discriminator_VGG_128_modified(3, 64, normalization_type='batch_normalization', activation_type='leakyrelu', mode='CNA');
    ### Load Discriminator: ###
    Discriminator_Network = load_Discriminator_from_checkpoint(models_folder, load_Discriminator_filename, Discriminator_Network)
    ### Discriminator Network to GPU: ###
    Discriminator_Network = Discriminator_to_GPU(Discriminator_Network)
    ### Discriminator to train mode: ###
    Discriminator_Network = Discriminator_to_train_mode(Discriminator_Network);
    ### Discriminator Optimizer: ###
    discriminator_learning_rate = 1e-4
    discriminator_hypergrad_lr = 5e-9
    optimizer_D = get_Discriminator_Optimizer(Discriminator_Network, learning_rate=discriminator_learning_rate, hypergrad_lr=discriminator_hypergrad_lr)

################################
### Generator Loss Function: ###
################################
DirectPixels_Loss_Function = nn.L1Loss()
# DirectPixels_Loss_Function = nn.SmoothL1Loss()  #TODO: decide with which to go with
FeatureExtractor_Loss_Function = nn.L1Loss()
Contextual_Loss_Function = Contextual_Loss(ContextualLoss_flag_L2_or_dotP)
GradientSensitive_Loss_Function = Gradient_Sensitive_Loss(lambda_GradientSensitive_low2high_ratio)
Gram_Loss_Function = Gram_Loss();
### Generator Loss Functions to GPU: ###
Generator_Loss_Functions_to_GPU()

#####################################
### Discriminator Loss Functions: ###
#####################################
random_alpha_tensor = torch.Tensor(1, 1, 1, 1).to(discriminator_device)
GAN_Validity_Loss_Function = GAN_validity_loss(Discriminator_Loss_Type, 1.0, 0.0);  # vanilla (BCEWithLogitsLoss), lsgan (MSELoss), wgan-gp (Wasserstein Loss With Gradient Penalty)
GradientPenalty_Loss_Function = GradientPenaltyLoss()  # why shouldn't there be a gradient penalty loss on the Generator as well?
Relativistic_GAN_Validity_Loss_Function = Relativistic_GAN_validity_loss();
### Discriminator Loss Functions to GPU: ###
Discriminator_Loss_Functions_to_GPU()
########################################################################################################################################



#### Klepto: Save whatever variables you want to be able to later get (probably mostly those which give description of the current Network & Hyperparameters and training parameters and such): ###
# save_filename_klepto = 'C:/Users\dkarl\PycharmProjects\dudykarl/loss_arrays_klepto2'
# whole_script_klepto_db = save_variables_to_klepto_file(save_filename_klepto,{'bla':bla});
# db = file_archive(save_filename_klepto+'.txt');
# db.load();
# db_keys_list = list(db.keys());
# db_values_list = list(db.values());




########################################################################################################################################
### Initialize Some Things For The Training Loop: ###

#(*). Initialize Discriminator Metrics:
number_of_correct_real_classifications = 0;
number_of_correct_fake_classifications = 0;
number_of_total_classifications = 0;

### Actually Train Generator: ###
torch.cuda.empty_cache()
#Get Layers Pointer For Tracking:
# Generator_layers_list = list(Generator_Network.children());
# hidden_states_example1 = list(Generator_layers_list[2].children())[0].hidden_states
# hidden_states_example2 = list(Generator_layers_list[3].children())[0].hidden_states

### LR Schedualer: ###
flag_use_LR_schedualer = True;
# lr_schedualer = lr_scheduler.MultiStepLR(optimizer_G, milestones=[500,1500,3000], gamma=0.1)
lr_schedualer = lr_scheduler.ReduceLROnPlateau(optimizer_G,
                                               mode='min',
                                               factor=learning_rate_lowering_factor,
                                               patience=learning_rate_lowering_patience_before_lowering,
                                               threshold=learning_rate_lowering_tolerance_factor_from_best_so_far,
                                               threshold_mode='rel')



### Residual Learning: ###
gaussian_blur_layer = get_gaussian_kernel(kernel_size=5, sigma=5, channels=3)  ####  Use the parameters relevant for the experiment!!@##@  ####
gaussian_blur_layer.to(Generator_device)

### Temp Correct: double_relu to no_activation: ###
# Generator_Network.up_concat1.conv_block_on_fused_inputs[2] = no_activation()
#############################################################################################################################################################################################################################








#######################################################################################################################################
#(*). Still Images
batch_size = 6*1; #TODO: lowered!
crop_size = 100
max_number_of_images = batch_size*1000000000
def toNp(pic):
    nppic = np.array(pic)
    return nppic
train_transform_PIL = transforms.Compose([
    transforms.RandomCrop(crop_size),
    transforms.Lambda(toNp),
    transforms.ToTensor(),
])
# images_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl/Imagenet'
images_folder = 'F:/NON_JPEG_IMAGES_BROKEN_INTO_CROPS\Images_CropSize_150'
# images_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl\Movie2'
train_dataset = ImageFolderRecursive_MaxNumberOfImages(images_folder, transform=train_transform_PIL, loader='PIL', max_number_of_images=max_number_of_images, min_crop_size=crop_size, extention='png')
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, pin_memory=True);  ### Mind the shuffle variable!@#!#%@#%@$%


# ### Show some dataloader images: ###
# max_number_of_batches = 10;
# tic()
# for batch_index, data_from_dataloader in enumerate(train_dataloader):
#     if batch_index > max_number_of_batches:
#         break;
#     imshow_torch(data_from_dataloader[0,:])
# toc()
######################################################################################################################################




######################################################################################################################################
### Loop Parameters: ###
#(*). Change Learning Rate If Wanted:
multiply_optimizer_learning_rate(optimizer_G,  lr_factor)  ### Changef learning rate  0.2*0.2*0.2*0.5*0.1
######################################################################################################################################




######################################################################################################################################
### Time Parametrs: ###
number_of_total_backward_steps_per_image = 5 #number of time steps before changing images batch
number_of_time_steps = 5; #number of time steps before doing a .backward()

### Time Weights: ###
flag_use_all_frames = True; # True=all frames (with below listed weights), False=only last frame
time_steps_weights = [1] * number_of_time_steps
time_steps_weights = time_steps_weights*number_of_total_backward_steps_per_image
step = 0.05
if step != 0:
    time_steps_weights += arange(0,step*number_of_total_backward_steps_per_image*number_of_time_steps,step)
# time_steps_weights[0] = 0


### Amount Of Training: ###
number_of_epochs = 100;
flag_stop_iterating = False;

### Initialize Loop Steps: ###
global_batch_step = 0;
global_step = 0
current_step = 0;
Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
Generator_Network.reset_hidden_states()

### Parameters: ###
max_noise_sigma = 0.1;


### Options For Optimizer Learning Rate and Stuff: ###
# multiply_optimizer_learning_rate(optimizer_G, 0)  ### Change learning rate
# multiply_optimizer_learning_rate(optimizer_G, 10)  ### Change learning rate
# optimizer_G = get_Generator_Optimizer(Generator_Network, learning_rate, hypergrad_lr);
# get_optimizer_learning_rate(optimizer_G)


### Loop Over The Epochs: ###
flag_stop_iterating = False
if flag_write_TB:
    # TB_writer = SummaryWriter('C:/Users\dkarl\PycharmProjects\dudykarl/TNR/TensorBoard/Generator_New_Noise_Model/' + Generator_checkpoint_prefix)
    TB_writer = SummaryWriter('C:/Users\dkarl\PycharmProjects\dudykarl/TNR/TensorBoard/Generator/' + Generator_checkpoint_prefix)
averaged_image = 0;
for current_epoch in arange(0, number_of_epochs):
    # tic()
    print(current_epoch)
    if flag_stop_iterating:
        break;

    for batch_index, data_from_dataloader in enumerate(train_dataloader):
        if (global_batch_step == max_total_number_of_batches or flag_stop_iterating==True) and flag_limit_number_of_iterations:
            flag_stop_iterating = True;
            break;
        global_batch_step += 1

        ###################################################################################################################################################################################################
        ########################################  Train Generator: #############################################
        torch.cuda.empty_cache()
        number_of_same_image_backward_steps = 0;
        number_of_time_steps_from_batch_start = 0;
        averaged_image = 0;
        print('New Batch')
        while data_from_dataloader.shape[0]==batch_size and flag_train_Generator and flag_stop_iterating==False and number_of_same_image_backward_steps<number_of_total_backward_steps_per_image and global_step >= number_of_initial_Discriminator_steps and ((flag_use_threshold_or_predetermined_GAN_policy == 'threshold' and current_generator_loss > Generator_loss_threshold) or (flag_use_threshold_or_predetermined_GAN_policy == 'predetermined' and global_step % discriminator_to_generator_training_steps == 0)):

            #############################################################################
            ### Get Proper Flags For Memory Cells: ###
            if number_of_same_image_backward_steps == 0:  # at first initialize with input_
                if flag_initialize_hidden_with_zero_or_input == 'zero':
                    reset_hidden_states_flags_list = [1] * batch_size;
                elif flag_initialize_hidden_with_zero_or_input == 'input':
                    reset_hidden_states_flags_list = [2] * batch_size;
            elif number_of_same_image_backward_steps > 0:
                reset_hidden_states_flags_list = [3] * batch_size;  # detach by default after every number_of_time_steps of graph accumulation
            #############################################################################


            #############################################################################
            ### Uptick Steps: ###
            number_of_same_image_backward_steps += 1;
            global_step += 1;
            #############################################################################


            #############################################################################
            Generator_Network = Generator_to_train_mode(Generator_Network)  # TODO: change to train_mode
            if flag_use_validity_loss:
                Discriminator_Network = Discriminator_to_train_mode(Discriminator_Network)  # TODO: change to eval mode
            torch.cuda.empty_cache()
            #############################################################################

            #############################################################################
            ### Zero Generator Grad & Reset Hidden States: ###
            torch.cuda.empty_cache()
            optimizer_G.zero_grad();
            total_generator_loss = 0
            #############################################################################



            ############## Go Through Time Steps: #########################
            tic()
            for current_time_step in arange(0, number_of_time_steps):

                ### Time-Shift & Put Noise in input tensor: ###
                # (1). Get Original Tensors:
                if len(data_from_dataloader.shape)==5:
                    #Time-Series
                    current_clean_tensor = data_from_dataloader[:, current_time_step, :, :, :].type(torch.FloatTensor)
                    current_noisy_tensor = data_from_dataloader[:, current_time_step, :, :, :].type(torch.FloatTensor)
                    current_clean_tensor = current_clean_tensor / 255;
                    current_noisy_tensor = current_noisy_tensor / 255;
                else:
                    #Still Image:
                    current_clean_tensor = data_from_dataloader.type(torch.FloatTensor)
                    current_noisy_tensor = data_from_dataloader.type(torch.FloatTensor)
                current_clean_tensor = current_clean_tensor.to(Generator_device);
                current_noisy_tensor = current_noisy_tensor.to(Generator_device);


                ### Add Noise: ###
                dshape = current_noisy_tensor.shape
                dshape = np.array(dshape)
                current_noisy_tensor = noise_RGB_LinearShotNoise_Torch(current_noisy_tensor, gain=noise_gain, flag_output_uint8_or_float='float')
                if flag_clip_noisy_tensor_after_noising:
                    current_noisy_tensor = current_noisy_tensor.clamp(0,1)


                ### Calculate Running Average: ###
                if current_time_step == 0 and number_of_same_image_backward_steps == 1: #very start of batch
                    print('initialize average image with noisy tensor')
                    averaged_image = current_noisy_tensor;
                else:
                    total_time_steps_this_batch = (number_of_same_image_backward_steps-1)*number_of_time_steps + current_time_step
                    alpha = total_time_steps_this_batch / (total_time_steps_this_batch+1)
                    averaged_image = (averaged_image*alpha + current_noisy_tensor*(1-alpha));



                ####################################################################################################################################################################
                ### Pass noisy input through network: ###
                current_noisy_tensor = Variable(current_noisy_tensor, requires_grad=False)
                output_tensor = Generator_Network(current_noisy_tensor, reset_hidden_states_flags_list)
                reset_hidden_states_flags_list = [0] * batch_size;  #In any case after whatever initial initialization signal -> do nothing and continue accumulating graph
                ####################################################################################################################################################################



                #############################################################################
                ### Get Target Images and Fake Images According to whether we're using residual learning: ###
                if flag_residual_learning:
                    if flag_learn_clean_or_running_average == 0:
                        real_images = current_clean_tensor - gaussian_blur_layer(current_noisy_tensor)  ### The Delta between clean and the blurred output
                    else:
                        real_images = averaged_image - gaussian_blur_layer(current_noisy_tensor)  ##TODO: generalize this with flag!!!!!
                    fake_images = output_tensor[:, 0:3, :, :];  # Delete - for ConvLSTM2D only
                else:
                    if flag_learn_clean_or_running_average == 0:
                        real_images = current_clean_tensor; #current_clean_tensor
                    else:
                        real_images = averaged_image;  ##TODO: generalize this with flag!!!
                    fake_images = output_tensor[:, 0:3, :, :];  # Delete - for ConvLSTM2D only
                #############################################################################



                #############################################################################
                ### Show real and fake images as the time steps accumulate!: ###
                # flag_show_generated_images_while_Generator_training = True
                if flag_show_generated_images_while_Generator_training:
                    if flag_residual_learning:
                        figure(2)
                        if flag_show_image_or_highpass_details == 0:  #Image
                            final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor)
                            final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor)
                        if flag_show_image_or_highpass_details == 1: #HighPass Details
                            final_real_images = real_images
                            final_fake_images = fake_images
                        L1_loss_now = nn.L1Loss()(final_real_images, final_fake_images).item()
                        subplot(1, 2, 1)
                        imshow(final_real_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                        title('real image')
                        subplot(1, 2, 2)
                        imshow(final_fake_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                        title('fake image, L1: ' + str(L1_loss_now))
                        pause(0.1);
                        matplotlib.pyplot.show(block=False)
                    else:
                        if flag_show_image_or_highpass_details == 1: #HighPass Details
                            final_real_images = real_images - gaussian_blur_layer(current_noisy_tensor)
                            final_fake_images = fake_images - gaussian_blur_layer(current_noisy_tensor)
                        if flag_show_image_or_highpass_details == 0: #Image
                            final_real_images = real_images;
                            final_fake_images = fake_images;
                        figure(1)
                        subplot(1,2,1)
                        imshow(final_real_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                        title('real image')
                        subplot(1,2,2)
                        imshow(final_fake_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                        title('fake image')
                        pause(0.1);
                        matplotlib.pyplot.show(block=False)
                #############################################################################


                ### Get Current Time-Step Weight: ###
                lambda_TimeStep = time_steps_weights[number_of_time_steps_from_batch_start]
                number_of_time_steps_from_batch_start += 1

                if flag_use_all_frames or current_time_step==number_of_time_steps-1:
                    ### Direct Pixels PixelWise (L1) Loss: ###
                    current_direct_pixels_loss = DirectPixels_Loss_Function(real_images, fake_images)
                    # current_direct_pixels_loss = nn.MSELoss()(real_images, fake_images)
                    if flag_use_direct_pixels_L1_loss:
                        current_generator_direct_pixels_loss = lambda_TimeStep * lambda_DirectPixels * current_direct_pixels_loss;
                        total_generator_loss += current_generator_direct_pixels_loss  # TODO: should i use total_loss += loss or total_loss+=loss.item()/loss.data()/loss.detach().....whatever....
                        torch.cuda.empty_cache()
                        running_average_image_L1 = nn.L1Loss()(averaged_image, real_images).item();

                        average_to_clean = nn.L1Loss()(averaged_image, current_clean_tensor).item();  # between average image and Clean
                        generated_to_clean = nn.L1Loss()(fake_images, current_clean_tensor).item();  # between clean image and Generated
                        generated_to_average = nn.L1Loss()(fake_images, averaged_image).item();  # between average image and Generated
                        generated_to_target = current_direct_pixels_loss.item()
                        running_average_image_L1 = generated_to_average

                        # #(1). Comparing to clean image all the time:
                        # print('Output L1: ' + str(current_generator_direct_pixels_loss.item()/lambda_TimeStep) + ', Average L1: ' + str(running_average_image_L1))

                        # (2). Comparing to running averaged image:
                        print('Fake-Clean: ' + str(generated_to_clean) + ', Fake-Average: ' + str(generated_to_average) + ', Average-Clean: ' + str(average_to_clean) + ', Fake-Target: ' + str(generated_to_target))


                    ### Direct Pixels Gram Loss: ###
                    if flag_use_direct_pixels_gram_loss:
                        current_generator_direct_pixels_gram_loss = lambda_TimeStep * lambda_DirectPixels_Gram * Gram_Loss_Function(real_images, fake_images)
                        total_generator_loss += current_generator_direct_pixels_gram_loss;
                        torch.cuda.empty_cache()

                    ### Direct Pixels Contextual Loss: ###
                    if flag_use_direct_pixels_contextual_loss:
                        direct_pixels_Contextual_patch_factor = 10;
                        real_patches = extract_patches_2D(real_images, [int(real_images.shape[2] / direct_pixels_Contextual_patch_factor), int(real_images.shape[3] / direct_pixels_Contextual_patch_factor)])
                        fake_patches = extract_patches_2D(fake_images, [int(fake_images.shape[2] / direct_pixels_Contextual_patch_factor), int(fake_images.shape[3] / direct_pixels_Contextual_patch_factor)])
                        current_generator_direct_pixels_contextual_loss = lambda_TimeStep * lambda_DirectPixels_Contextual * Contextual_Loss_Function(real_patches, fake_patches);
                        total_generator_loss += current_generator_direct_pixels_contextual_loss;
                        torch.cuda.empty_cache()

                    ### Gradient Sensitive Loss: ###
                    if flag_use_gradient_sensitive_loss:
                        current_generator_direct_pixels_gradient_sensitive_loss = lambda_TimeStep * lambda_DirectPixels_GradientSensitive * GradientSensitive_Loss_Function(real_images, fake_images);
                        total_generator_loss += current_generator_direct_pixels_gradient_sensitive_loss;
                        torch.cuda.empty_cache()

                    ### Feature Extractor Loss: ###
                    if flag_use_feature_extractor_direct_loss or flag_use_feature_extractor_contextual_loss or flag_use_feature_extractor_gram_loss:
                        # Get a list of returned outputs:
                        real_images_features_extracted = netF(real_images.to(netF_device))  # TODO: at first there was .detach() here...because i'm changing the structure i'm removing it to below logic
                        fake_images_features_extracted = netF(fake_images.to(netF_device))
                        # Loop over outputs list and compare features with L1 and Contextual Loss:
                        for i in arange(len(real_images_features_extracted)):
                            if flag_use_feature_extractor_direct_loss:
                                current_generator_feature_extractor_loss = lambda_TimeStep * lambda_Features_Perceptual * FeatureExtractor_Loss_Function(fake_images_features_extracted[i], real_images_features_extracted[i].detach())
                                total_generator_loss += current_generator_feature_extractor_loss.to(Generator_device)
                                torch.cuda.empty_cache()
                            if flag_use_feature_extractor_contextual_loss:
                                current_generator_feature_contextual_loss = lambda_TimeStep * lambda_Features_Contextual * Contextual_Loss_Function(fake_images_features_extracted[i], real_images_features_extracted[i].detach())
                                total_generator_loss += current_generator_feature_contextual_loss.to(Generator_device)  # TODO: does it make sense to do a "loss_device"?
                                torch.cuda.empty_cache()
                            if flag_use_feature_extractor_gram_loss:
                                current_generator_feature_extractor_gram_loss = lambda_TimeStep * lambda_Features_Gram * Gram_Loss_Function(fake_images_features_extracted[i], real_images_features_extracted[i].detach());
                                total_generator_loss += current_generator_feature_extractor_gram_loss;
                                torch.cuda.empty_cache()

                            torch.cuda.empty_cache()

                    ### GAN Discriminator Model Loss: ###
                    if flag_use_validity_loss:
                        real_images = real_images.to(discriminator_device);
                        fake_images = fake_images.to(discriminator_device);
                        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(discriminator_device)
                        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(discriminator_device)
                        if flag_use_fp16: mean = mean.half()  # TODO: fp16 delete if wanted
                        if flag_use_fp16: std = std.half()  # TODO: fp16 delete if wanted
                        # TODO: remember to use normalization if so wanted...for instance if i use transfer learning on a pretrained imagenet model i would want to do something like (real_images-mean)/std
                        D_output_real_image_batch_validity = Discriminator_Network(real_images).detach()
                        D_output_fake_image_batch_validity = Discriminator_Network(fake_images)
                        current_generator_validity_loss = lambda_TimeStep * lambda_GAN_validity_loss * Relativistic_GAN_Validity_Loss_Function(GAN_Validity_Loss_Function, D_output_fake_image_batch_validity, D_output_real_image_batch_validity)
                        total_generator_loss += current_generator_validity_loss.to(Generator_device)
                        number_of_correct_real_classifications = ((D_output_real_image_batch_validity - torch.mean(D_output_fake_image_batch_validity)) > 0).sum(dim=0).cpu().numpy()
                        number_of_correct_fake_classifications = ((D_output_fake_image_batch_validity - torch.mean(D_output_real_image_batch_validity)) < 0).sum(dim=0).cpu().numpy()
                        number_of_total_classifications = D_output_real_image_batch_validity.shape[0]
                        current_accuracy_real_images = number_of_correct_real_classifications / number_of_total_classifications
                        current_accuracy_fake_images = number_of_correct_fake_classifications / number_of_total_classifications
                        torch.cuda.empty_cache()
            ### END Of Generator Loss Accumulation! ###




            ###############################################################################
            ### Adjust Learning Rate According To Current Loss: ###

            ### Update LR Using LR Schedualer: ###
            if flag_use_LR_schedualer:
                lr_schedualer.step(total_generator_loss)

            ### Get Generator Learning Rate: ###
            #TODO: generalize to multiple parameter groups <-> multiple learning rates
            if type(optimizer_G.param_groups[0]['lr']) == torch.Tensor:
                Generator_lr = optimizer_G.param_groups[0]['lr'].cpu().item();
            elif type(optimizer_G.param_groups[0]['lr']) == float:
                Generator_lr = optimizer_G.param_groups[0]['lr'];

            ### If Learning Rate changed do some stuff: ###
            if Generator_lr < Generator_lr_previous:
                print('CHANGED LEEARNING RATE!!!!')
                if flag_change_optimizer_betas:
                    for i, param_group in enumerate(optimizer_G.param_groups):
                        param_group['betas'] = tuple([1 - (1 - b) * 0.1 for b in param_group['betas']])  # beta_new = 0.1*beta_old + 0.9
            Generator_lr_previous = Generator_lr;

            ### If Learning Rate falls below minimum then spike it up: ###
            if Generator_lr < minimum_learning_rate:
                minimum_learning_rate_counter += 1;
                #If we're below the minimum learning rate than it's time to jump the LR up to try and get to a lower local minima:

                ### Initialize Learning Rate (later to be subject to lr scheduling): ###
                learning_rate = init_learning_rate * (initial_spike_learning_rate_reduction_multiple ** minimum_learning_rate_counter)  # TODO: make hyper-parameter  and instead of adjusting the initial LR at the start of each meta_epoch simply adjust it after each "meta-epoch" as i define it which would be dynamic
                set_optimizer_learning_rate(optimizer_G, learning_rate)

                ### Initialize Learning Rate Scheduler: ###
                lr_schedualer = lr_scheduler.ReduceLROnPlateau(optimizer_G,
                                                               mode='min',
                                                               factor=learning_rate_lowering_factor,
                                                               patience=learning_rate_lowering_patience_before_lowering,
                                                               threshold=learning_rate_lowering_tolerance_factor_from_best_so_far,
                                                               threshold_mode='rel')
            #############################################################################


            #############################################################################
            ### Back-Propagate: ###
            torch.cuda.empty_cache()
            total_generator_loss.backward()

            ### Clip Gradient Norm: ###
            if flag_use_gradient_clip_valueping:
                if flag_clip_network_or_center == 'network':
                    if flag_clip_direct_grad_or_norm == 'norm':
                        total_norm = torch.nn.utils.clip_grad_norm_(Generator_Network.parameters(), gradient_clip_value)
                    elif flag_clip_direct_grad_or_norm == 'direct':
                        total_norm = torch.nn.utils.clip_grad_value_(Generator_Network.parameters(), gradient_clip_value)
                elif flag_clip_network_or_center == 'center':
                    if flag_clip_direct_grad_or_norm == 'norm':
                        total_norm = torch.nn.utils.clip_grad_norm_(Generator_Network.center.parameters(), gradient_clip_value)
                    elif flag_clip_direct_grad_or_norm == 'direct':
                        total_norm = torch.nn.utils.clip_grad_value_(Generator_Network.center.parameters(), gradient_clip_value)
                # (*). Update norm moving average: (*)#
                #TODO: make sure gradient_Clip_value updates because maybe not...
                gradient_clip_value = ema_object(total_norm, gradient_clip_value)

            print('Total Gen Loss: ' + str(total_generator_loss.item()) + ', Generator LR: ' + str(Generator_lr))

            ### Update Weights: ###
            torch.cuda.empty_cache();
            optimizer_G.step();
            torch.cuda.empty_cache();
            #############################################################################



            #############################################################################
            # Keep track of Stuff i want to keep track of:
            if flag_write_TB:
                flag_TB_average_image = True;
                if flag_TB_average_image:
                    TB_writer.add_scalar('Generator/Average_DirectPixel_Loss', running_average_image_L1, global_step + Generator_checkpoint_step)
                if flag_use_direct_pixels_L1_loss:
                    TB_writer.add_scalar('Generator/DirectPixel_Loss', current_generator_direct_pixels_loss.item()/lambda_TimeStep, global_step+Generator_checkpoint_step)
                    # TB_writer.add_scalar('Generator/DirectPixel_Loss', average_to_clean, global_step+Generator_checkpoint_step)
                if flag_use_feature_extractor_direct_loss:
                    TB_writer.add_scalar('Generator/FeatureExtractor_Loss', current_generator_feature_extractor_loss.item()/lambda_TimeStep, global_step+Generator_checkpoint_step)
                if flag_use_feature_extractor_contextual_loss:
                    TB_writer.add_scalar('Generator/FeatureExtractor_Contextual_Los', current_generator_feature_contextual_loss.item()/lambda_TimeStep, global_step+Generator_checkpoint_step)
                if flag_use_direct_pixels_contextual_loss:
                    TB_writer.add_scalar('Generator/DirectPixels_Contextual_Loss', current_generator_direct_pixels_contextual_loss.item()/lambda_TimeStep, global_step+Generator_checkpoint_step)
                if flag_use_gradient_sensitive_loss:
                    TB_writer.add_scalar('Generator/GradientSensitive_Loss', current_generator_direct_pixels_gradient_sensitive_loss.item()/lambda_TimeStep, global_step+Generator_checkpoint_step)
                if flag_use_validity_loss:
                    TB_writer.add_scalar('Discriminator/RealImages_Accuracy', current_accuracy_real_images, global_step+Generator_checkpoint_step)
                    TB_writer.add_scalar('Discriminator/FakeImages_Accuracy', current_accuracy_fake_images, global_step+Generator_checkpoint_step)
                TB_writer.add_scalar('Generator/Learning_Rate', Generator_lr, global_step+Generator_checkpoint_step)
            #############################################################################


            #############################################################################
            # Get Print String:
            print_string = 'G Step ' + scientific_notation(global_step) + ', '
            if flag_use_validity_loss:
                print_string += 'Val Loss: ' + scientific_notation(current_generator_validity_loss.item()) + ', '
                print_string += 'Fake Val Acc: ' + str(float(current_accuracy_fake_images)) + ', '
                print_string += 'Real Val Acc: ' + str(float(current_accuracy_real_images)) + ', '
            if flag_use_direct_pixels_L1_loss:
                print_string += 'Direct L1: ' + scientific_notation(current_generator_direct_pixels_loss.item()) + ', '
            if flag_use_feature_extractor_direct_loss:
                print_string += 'Features L1: ' + scientific_notation(current_generator_feature_extractor_loss.item()) + ', '
            if flag_use_direct_pixels_contextual_loss:
                print_string += 'Direct Contextual:' + scientific_notation(current_generator_direct_pixels_contextual_loss.item()) + ', '
            if flag_use_feature_extractor_contextual_loss:
                print_string += 'Features Contextual:' + scientific_notation(current_generator_feature_contextual_loss.item()) + ', '
            if flag_use_gradient_sensitive_loss:
                print_string += 'GradientSensitive Loss: ' + scientific_notation(current_generator_direct_pixels_gradient_sensitive_loss.item()) + ', '
            if flag_use_direct_pixels_gram_loss:
                print_string += 'Direct Gram: ' + scientific_notation(current_generator_direct_pixels_gram_loss.item()) + ', '
            if flag_use_feature_extractor_gram_loss:
                print_string += 'Features Gram: ' + scientific_notation(current_generator_feature_extractor_gram_loss.item()) + ', '
            print_string += 'Gen lr: ' + scientific_notation(Generator_lr) + ', '  # encoder lr
            toc(print_string)
            #############################################################################







            #################################################################################################################################################################################################################
            ##########################################  Save Models:  #################################
            if global_step % save_models_frequency == 0 or (current_epoch == number_of_epochs - 1 and batch_index + batch_size >= len(train_dataset)):
                ### Save Generator and Discriminator: ###
                print('Save Generator and Discriminator Models:')
                print(Generator_Network); #Output Network Architecture for me to be able to keep track of what's going on when having multiple experiments simultaneously
                print(Generator_checkpoint_prefix)
                current_Generator_iteration_string = 'Step' + str(global_step+Generator_checkpoint_step) + '_GenLR_' + scientific_notation(Generator_lr).replace('.','d').replace('-','m')
                current_Discriminator_iteration_string = 'Step' + str(global_step)
                basic_Generator_filename = Generator_checkpoint_prefix + '_' + Generator_checkpoint_postfix.split('.')[0] + '_' + current_Generator_iteration_string
                basic_Discriminator_filename = Discriminator_checkpoint_prefix + '_' + Discriminator_checkpoint_postfix.split('.')[0] + '_' + current_Discriminator_iteration_string
                ### Save Generator: ###
                save_Generator_parts_to_checkpoint(Generator_Network, Generator_checkpoint_folder, basic_Generator_filename, flag_save_dict_or_whole='both')
                ### Save Discriminator: ###
                if flag_use_validity_loss:
                    save_Discriminator_to_checkpoint(Discriminator_Network, Discriminator_checkpoint_folder, basic_Discriminator_filename, flag_save_dict_or_whole='both')

                ### Plot Gradients: ###
                #TODO: add this to TensorBoard - understand how to add figures to tensorboard besides numpy arrays
                if flag_plot_grad_flow:
                    plot_grad_flow(Generator_Network)
                    pause(0.1)


                ### Go Through Evaluation Set and Get What i Want (for instance images): ###
                # Evaluate_Model(Generator_Network);




    toc('End of Epoch: ')

if flag_write_TB:
    TB_writer.export_scalars_to_json("./all_scalars.json")
    TB_writer.close()
#############################################################################################################################################################################################################################






































# #############################################################################################################################################################################################################################
# ### Show Still Image Results: ###
#
#
#
#
# #(1). Start Recording Video:
# image_index = 1;
# fps = 1.0
# flag_save_videos = True;
# flag_show_results = True;
# flag_save_video_or_images = 'both' #'images' / 'videos' / 'both'
#
# crop_size = 20000;
# created_video_filename_prefix = load_Generator_filename.split('.')[0]
# super_folder_to_save_result_videos = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos Still Images/'
# sub_folder_to_save_current_results_videos_ALL_DETAILS = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos Still Images/' + created_video_filename_prefix + '/ALL_DETAILS';
# sub_folder_to_save_current_results_videos_HIGHPASS = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos Still Images/' + created_video_filename_prefix + '/HIGHPASS';
# if not os.path.exists(sub_folder_to_save_current_results_videos_ALL_DETAILS):
#     os.makedirs(sub_folder_to_save_current_results_videos_ALL_DETAILS)
#     if not os.path.exists(sub_folder_to_save_current_results_videos_HIGHPASS):
#         os.makedirs(sub_folder_to_save_current_results_videos_HIGHPASS)
# video_postfix_type = '.mpeg'
# #ALL DETAILS:
# created_video_filename_original_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'original' + video_postfix_type)
# created_video_filename_noisy_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'noisy' + video_postfix_type)
# created_video_filename_clean_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'clean' + video_postfix_type)
# created_video_filename_Concat_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'Concat' + video_postfix_type)
# #HIGHPASS:
# created_video_filename_original_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'original' + video_postfix_type)
# created_video_filename_noisy_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'noisy' + video_postfix_type)
# created_video_filename_clean_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'clean' + video_postfix_type)
# created_video_filename_Concat_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'Concat' + video_postfix_type)
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # Be sure to use lower case  MP42
#
# if flag_save_video_or_images == 'images' or flag_save_video_or_images == 'both':
#     #ALL DETAILS:
#     original_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'original')
#     noisy_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'noisy')
#     clean_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'clean')
#     Concat_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'Concat')
#     if not os.path.exists(original_video_images_folder_ALL_DETAILS):
#         os.makedirs(original_video_images_folder_ALL_DETAILS)
#     if not os.path.exists(noisy_video_images_folder_ALL_DETAILS):
#         os.makedirs(noisy_video_images_folder_ALL_DETAILS)
#     if not os.path.exists(clean_video_images_folder_ALL_DETAILS):
#         os.makedirs(clean_video_images_folder_ALL_DETAILS)
#     if not os.path.exists(Concat_video_images_folder_ALL_DETAILS):
#         os.makedirs(Concat_video_images_folder_ALL_DETAILS)
#     #HIGHPASS:
#     original_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'original')
#     noisy_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'noisy')
#     clean_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'clean')
#     Concat_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'Concat')
#     if not os.path.exists(original_video_images_folder_HIGHPASS):
#         os.makedirs(original_video_images_folder_HIGHPASS)
#     if not os.path.exists(noisy_video_images_folder_HIGHPASS):
#         os.makedirs(noisy_video_images_folder_HIGHPASS)
#     if not os.path.exists(clean_video_images_folder_HIGHPASS):
#         os.makedirs(clean_video_images_folder_HIGHPASS)
#     if not os.path.exists(Concat_video_images_folder_HIGHPASS):
#         os.makedirs(Concat_video_images_folder_HIGHPASS)
#
#
# ### Time Parametrs: ###
# number_of_total_backward_steps_per_image = 2 #number of time steps before changing images batch
# number_of_time_steps = 5; #number of time steps before doing a .backward()
#
#
# ### Amount Of Training: ###
# number_of_epochs = 1;
# max_total_number_of_batches = 2;  #10000
# flag_limit_number_of_iterations = True;
# # flag_limit_number_of_iterations = False;
# flag_stop_iterating = False;
#
# ### Initialize Loop Steps: ###
# global_batch_step = 0;
# number_of_time_steps_so_far = 0
# current_step = 0;
# Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
#
# ### Parameters: ###
# max_noise_sigma = 0.1;
#
# ### Show Parameters: ###
# flag_show_generated_images_while_Generator_training = True;
# flag_residual_learning = False;  #### MAKE SURE THIS FITS THE WAY THE ORIGINAL MODEL WAS TRAINED (TODO: maybe use a klepto file to save all those stuff i want to remember!!@#@?!#?!)
# flag_show_image_or_highpass_details = 0; # 0=image, 1=highpass details
# flag_learn_clean_or_running_average = 0;
# flag_use_all_frames = True
#
# for batch_index, data_from_dataloader in enumerate(train_dataloader):
#     if (global_batch_step == max_total_number_of_batches or flag_stop_iterating==True) and flag_limit_number_of_iterations:
#         flag_stop_iterating = True;
#         break;
#     global_batch_step += 1
#
#
#     ### At Very First Time Step Only: ###
#     if global_batch_step == 1:
#         batch_size, number_of_channels, frame_height, frame_width = data_from_dataloader.shape
#         crop_size_width = min(crop_size, frame_width);
#         crop_size_height = min(crop_size, frame_height);
#         start_index_height = random_integers(0,frame_height - crop_size_height);
#         start_index_width = random_integers(0,frame_width - crop_size_width)
#         if flag_save_videos:
#             #ALL DETAILS:
#             video_writer_original_ALL_DETAILS = cv2.VideoWriter(created_video_filename_original_ALL_DETAILS, fourcc, fps, (frame_width*1, frame_height))
#             video_writer_noisy_ALL_DETAILS = cv2.VideoWriter(created_video_filename_noisy_ALL_DETAILS, fourcc, fps, (frame_width * 1, frame_height))
#             video_writer_clean_ALL_DETAILS = cv2.VideoWriter(created_video_filename_clean_ALL_DETAILS, fourcc, fps, (frame_width * 1, frame_height))
#             video_writer_Concat_ALL_DETAILS = cv2.VideoWriter(created_video_filename_Concat_ALL_DETAILS, fourcc, fps, (frame_width * 3, frame_height))
#             #HIGHPASS:
#             video_writer_original_HIGHPASS = cv2.VideoWriter(created_video_filename_original_HIGHPASS, fourcc, fps, (frame_width * 1, frame_height))
#             video_writer_noisy_HIGHPASS = cv2.VideoWriter(created_video_filename_noisy_HIGHPASS, fourcc, fps, (frame_width * 1, frame_height))
#             video_writer_clean_HIGHPASS = cv2.VideoWriter(created_video_filename_clean_HIGHPASS, fourcc, fps, (frame_width * 1, frame_height))
#             video_writer_Concat_HIGHPASS = cv2.VideoWriter(created_video_filename_Concat_HIGHPASS, fourcc, fps, (frame_width * 3, frame_height))
#
#
#     ###################################################################################################################################################################################################
#     ########################################  Train Generator: #############################################
#     torch.cuda.empty_cache()
#     number_of_same_image_backward_steps = 0;
#     number_of_time_steps_from_batch_start = 0;
#     averaged_image = 0;
#     print('New Batch')
#     while data_from_dataloader.shape[0]==batch_size and flag_train_Generator and flag_stop_iterating==False and number_of_same_image_backward_steps<number_of_total_backward_steps_per_image:
#         ### Get Generator to train mode and Discriminator to eval mode:
#         if number_of_same_image_backward_steps == 0:  # at first initialize with input_
#             if flag_initialize_hidden_with_zero_or_input == 'zero':
#                 reset_hidden_states_flags_list = [1] * batch_size;
#             elif flag_initialize_hidden_with_zero_or_input == 'input':
#                 reset_hidden_states_flags_list = [2] * batch_size;
#         elif number_of_same_image_backward_steps > 0:
#             reset_hidden_states_flags_list = [3] * batch_size;  # detach by default after every number_of_time_steps of graph accumulation
#         ###                                                         ###
#
#         # if number_of_time_steps_so_far == 2:
#         #     flag_stop_iterating = True;
#         #     break;
#         Generator_Network = Generator_to_eval_mode(Generator_Network)
#         if flag_use_validity_loss:
#             Discriminator_Network = Discriminator_to_train_mode(Discriminator_Network)  # TODO: change to eval mode
#         torch.cuda.empty_cache()
#
#         ### Zero Generator Grad & Reset Hidden States: ###
#         torch.cuda.empty_cache()
#         total_generator_loss = 0
#
#         ### Decide On Noise Level: ###
#         noise_sigma = max_noise_sigma;
#
#
#         ############## Go Through Time Steps: #########################
#         tic()
#         number_of_same_image_backward_steps += 1;
#         for current_time_step in arange(0, number_of_time_steps):
#             number_of_time_steps_from_batch_start += 1
#             number_of_time_steps_so_far += 1;
#
#             ### Time-Shift & Put Noise in input tensor: ###
#             # (1). Get Original Tensors:
#             if len(data_from_dataloader.shape)==5:
#                 #Time-Series
#                 current_clean_tensor = data_from_dataloader[:, current_time_step, :, :, :].type(torch.FloatTensor)
#                 current_noisy_tensor = data_from_dataloader[:, current_time_step, :, :, :].type(torch.FloatTensor)
#                 current_clean_tensor = current_clean_tensor / 255;
#                 current_noisy_tensor = current_noisy_tensor / 255;
#             else:
#                 #Still Image:
#                 current_clean_tensor = data_from_dataloader.type(torch.FloatTensor)
#                 current_noisy_tensor = data_from_dataloader.type(torch.FloatTensor)
#             current_clean_tensor = current_clean_tensor.to(Generator_device);
#             current_noisy_tensor = current_noisy_tensor.to(Generator_device);
#
#             # (3). Add Noise:
#             dshape = current_noisy_tensor.shape
#             dshape = np.array(dshape)
#             # current_noisy_tensor = current_noisy_tensor + torch.Tensor(randn(dshape[0], dshape[1], dshape[2], dshape[3])).to(Generator_device) * noise_sigma  #Additive Noise
#             current_noisy_tensor_multiplicative_noise = current_noisy_tensor.mean(1,True) * noise_sigma * torch.Tensor(randn(dshape[0], dshape[1], dshape[2], dshape[3])).to(Generator_device)
#             current_noisy_tensor = current_noisy_tensor + current_noisy_tensor_multiplicative_noise  #Multiplicative Noise
#
#             if current_time_step == 0 and number_of_same_image_backward_steps == 1: #very start of batch
#                 print('initialize average image with noisy tensor')
#                 averaged_image = current_noisy_tensor;
#             else:
#                 total_time_steps_this_batch = (number_of_same_image_backward_steps-1)*number_of_time_steps + current_time_step
#                 alpha = total_time_steps_this_batch / (total_time_steps_this_batch+1)
#                 averaged_image = (averaged_image*alpha + current_noisy_tensor*(1-alpha));
#
#
#
#             ### Pass noisy input through network: ###
#             current_noisy_tensor = Variable(current_noisy_tensor, requires_grad=False)
#             output_tensor = Generator_Network(current_noisy_tensor, reset_hidden_states_flags_list)
#             reset_hidden_states_flags_list = [0] * batch_size;  #In any case after whatever initial initialization signal -> do nothing and continue accumulating graph
#
#             ### Rename For Easier Handling: ###
#             if flag_residual_learning:
#                 if flag_learn_clean_or_running_average == 0:
#                     real_images = current_clean_tensor - gaussian_blur_layer(current_noisy_tensor)  ### The Delta between clean and the blurred output
#                 else:
#                     real_images = averaged_image - gaussian_blur_layer(current_noisy_tensor)  ##TODO: generalize this with flag!!!!!
#                 fake_images = output_tensor[:, 0:3, :, :];  # Delete - for ConvLSTM2D only
#             else:
#                 if flag_learn_clean_or_running_average == 0:
#                     real_images = current_clean_tensor; #current_clean_tensor
#                 else:
#                     real_images = averaged_image;  ##TODO: generalize this with flag!!!
#                 fake_images = output_tensor[:, 0:3, :, :];  # Delete - for ConvLSTM2D only
#
#
#
#             ### Get Current Time-Step Weight: ###
#             lambda_TimeStep = 1
#
#             if flag_use_all_frames or current_time_step==number_of_time_steps-1:
#                 ### Direct Pixels PixelWise (L1) Loss: ###
#                 current_direct_pixels_loss = DirectPixels_Loss_Function(real_images, fake_images)
#                 # current_direct_pixels_loss = nn.MSELoss()(real_images, fake_images)
#                 if flag_use_direct_pixels_L1_loss:
#                     current_generator_direct_pixels_loss = lambda_TimeStep * lambda_DirectPixels * current_direct_pixels_loss;
#                     total_generator_loss += current_generator_direct_pixels_loss  # TODO: should i use total_loss += loss or total_loss+=loss.item()/loss.data()/loss.detach().....whatever....
#                     torch.cuda.empty_cache()
#                     running_average_image_L1 = nn.L1Loss()(averaged_image, real_images).item();
#
#                     average_to_clean = nn.L1Loss()(averaged_image, current_clean_tensor).item();  # between average image and Clean
#                     generated_to_clean = nn.L1Loss()(fake_images, current_clean_tensor).item();  # between clean image and Generated
#                     generated_to_average = nn.L1Loss()(fake_images, averaged_image).item();  # between average image and Generated
#                     generated_to_target = current_direct_pixels_loss.item()
#                     running_average_image_L1 = generated_to_average
#
#                     # #(1). Comparing to clean image all the time:
#                     # print('Output L1: ' + str(current_generator_direct_pixels_loss.item()/lambda_TimeStep) + ', Average L1: ' + str(running_average_image_L1))
#
#                     # (2). Comparing to running averaged image:
#                     print('Fake-Clean: ' + str(generated_to_clean) + ', Fake-Average: ' + str(generated_to_average) + ', Average-Clean: ' + str(average_to_clean) + ', Fake-Target: ' + str(generated_to_target))
#
#
#                 ### Direct Pixels Gram Loss: ###
#                 if flag_use_direct_pixels_gram_loss:
#                     current_generator_direct_pixels_gram_loss = lambda_TimeStep * lambda_DirectPixels_Gram * Gram_Loss_Function(real_images, fake_images)
#                     total_generator_loss += current_generator_direct_pixels_gram_loss;
#                     torch.cuda.empty_cache()
#
#                 ### Direct Pixels Contextual Loss: ###
#                 if flag_use_direct_pixels_contextual_loss:
#                     direct_pixels_Contextual_patch_factor = 10;
#                     real_patches = extract_patches_2D(real_images, [int(real_images.shape[2] / direct_pixels_Contextual_patch_factor), int(real_images.shape[3] / direct_pixels_Contextual_patch_factor)])
#                     fake_patches = extract_patches_2D(fake_images, [int(fake_images.shape[2] / direct_pixels_Contextual_patch_factor), int(fake_images.shape[3] / direct_pixels_Contextual_patch_factor)])
#                     current_generator_direct_pixels_contextual_loss = lambda_TimeStep * lambda_DirectPixels_Contextual * Contextual_Loss_Function(real_patches, fake_patches);
#                     total_generator_loss += current_generator_direct_pixels_contextual_loss;
#                     torch.cuda.empty_cache()
#
#                 ### Gradient Sensitive Loss: ###
#                 if flag_use_gradient_sensitive_loss:
#                     current_generator_direct_pixels_gradient_sensitive_loss = lambda_TimeStep * lambda_DirectPixels_GradientSensitive * GradientSensitive_Loss_Function(real_images, fake_images);
#                     total_generator_loss += current_generator_direct_pixels_gradient_sensitive_loss;
#                     torch.cuda.empty_cache()
#
#                 ### Feature Extractor Loss: ###
#                 if flag_use_feature_extractor_direct_loss or flag_use_feature_extractor_contextual_loss or flag_use_feature_extractor_gram_loss:
#                     # Get a list of returned outputs:
#                     real_images_features_extracted = netF(real_images.to(netF_device))  # TODO: at first there was .detach() here...because i'm changing the structure i'm removing it to below logic
#                     fake_images_features_extracted = netF(fake_images.to(netF_device))
#                     # Loop over outputs list and compare features with L1 and Contextual Loss:
#                     for i in arange(len(real_images_features_extracted)):
#                         if flag_use_feature_extractor_direct_loss:
#                             current_generator_feature_extractor_loss = lambda_TimeStep * lambda_Features_Perceptual * FeatureExtractor_Loss_Function(fake_images_features_extracted[i], real_images_features_extracted[i].detach())
#                             total_generator_loss += current_generator_feature_extractor_loss.to(Generator_device)
#                             torch.cuda.empty_cache()
#                         if flag_use_feature_extractor_contextual_loss:
#                             current_generator_feature_contextual_loss = lambda_TimeStep * lambda_Features_Contextual * Contextual_Loss_Function(fake_images_features_extracted[i], real_images_features_extracted[i].detach())
#                             total_generator_loss += current_generator_feature_contextual_loss.to(Generator_device)  # TODO: does it make sense to do a "loss_device"?
#                             torch.cuda.empty_cache()
#                         if flag_use_feature_extractor_gram_loss:
#                             current_generator_feature_extractor_gram_loss = lambda_TimeStep * lambda_Features_Gram * Gram_Loss_Function(fake_images_features_extracted[i], real_images_features_extracted[i].detach());
#                             total_generator_loss += current_generator_feature_extractor_gram_loss;
#                             torch.cuda.empty_cache()
#
#                         torch.cuda.empty_cache()
#
#                 ### GAN Discriminator Model Loss: ###
#                 if flag_use_validity_loss:
#                     real_images = real_images.to(discriminator_device);
#                     fake_images = fake_images.to(discriminator_device);
#                     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(discriminator_device)
#                     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(discriminator_device)
#                     if flag_use_fp16: mean = mean.half()  # TODO: fp16 delete if wanted
#                     if flag_use_fp16: std = std.half()  # TODO: fp16 delete if wanted
#                     # TODO: remember to use normalization if so wanted...for instance if i use transfer learning on a pretrained imagenet model i would want to do something like (real_images-mean)/std
#                     D_output_real_image_batch_validity = Discriminator_Network(real_images).detach()
#                     D_output_fake_image_batch_validity = Discriminator_Network(fake_images)
#                     current_generator_validity_loss = lambda_TimeStep * lambda_GAN_validity_loss * Relativistic_GAN_Validity_Loss_Function(GAN_Validity_Loss_Function, D_output_fake_image_batch_validity, D_output_real_image_batch_validity)
#                     total_generator_loss += current_generator_validity_loss.to(Generator_device)
#                     number_of_correct_real_classifications = ((D_output_real_image_batch_validity - torch.mean(D_output_fake_image_batch_validity)) > 0).sum(dim=0).cpu().numpy()
#                     number_of_correct_fake_classifications = ((D_output_fake_image_batch_validity - torch.mean(D_output_real_image_batch_validity)) < 0).sum(dim=0).cpu().numpy()
#                     number_of_total_classifications = D_output_real_image_batch_validity.shape[0]
#                     current_accuracy_real_images = number_of_correct_real_classifications / number_of_total_classifications
#                     current_accuracy_fake_images = number_of_correct_fake_classifications / number_of_total_classifications
#                     torch.cuda.empty_cache()
#
#
#                 ### Show Results: ###
#                 if flag_show_generated_images_while_Generator_training:
#                     if flag_residual_learning:
#                         figure(2)
#                         if flag_show_image_or_highpass_details == 0:  # Image
#                             final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor)
#                             final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor)
#                         if flag_show_image_or_highpass_details == 1:  # HighPass Details
#                             final_real_images = real_images
#                             final_fake_images = fake_images
#                         L1_loss_now = nn.L1Loss()(final_real_images, final_fake_images).item()
#                         subplot(1, 2, 1)
#                         imshow(final_real_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
#                         title('real image')
#                         subplot(1, 2, 2)
#                         imshow(final_fake_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
#                         title('fake image, L1: ' + str(L1_loss_now))
#                         pause(0.1);
#                         matplotlib.pyplot.show(block=False)
#                     else:
#                         if flag_show_image_or_highpass_details == 1:  # HighPass Details
#                             final_real_images = real_images - gaussian_blur_layer(current_noisy_tensor)
#                             final_fake_images = fake_images - gaussian_blur_layer(current_noisy_tensor)
#
#                         if flag_show_image_or_highpass_details == 0:  # Image
#                             final_real_images = real_images;
#                             final_fake_images = fake_images;
#                         figure(1)
#                         subplot(1, 2, 1)
#                         imshow(final_real_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
#                         title('real image')
#                         subplot(1, 2, 2)
#                         imshow(final_fake_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
#                         title('fake image')
#                         pause(0.1);
#                         matplotlib.pyplot.show(block=False)
#
#
#                 ### Loop to show both all details and highpass: ###
#                 for i in arange(2):
#                     flag_show_image_or_highpass_details = i;
#
#                     ### Save Results: ###
#                     ### Turn to uint8 numpy: ###
#                     if flag_residual_learning:
#                         if flag_show_image_or_highpass_details == 0:  # Image
#                             final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor)
#                             final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor)
#                             final_noisy_images = current_noisy_tensor
#                         if flag_show_image_or_highpass_details == 1:  # HighPass Details
#                             final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor) - gaussian_blur_layer(current_clean_tensor)
#                             final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor) - gaussian_blur_layer(output_tensor[:,0:3,:,:])
#                             final_noisy_images = current_noisy_tensor - gaussian_blur_layer(current_noisy_tensor)
#                     else:
#                         if flag_show_image_or_highpass_details == 1:  # HighPass Details
#                             final_real_images = real_images - gaussian_blur_layer(real_images)
#                             final_fake_images = fake_images - gaussian_blur_layer(fake_images)
#                             final_noisy_images = current_noisy_tensor - gaussian_blur_layer(current_noisy_tensor)
#                         if flag_show_image_or_highpass_details == 0:  # Image
#                             final_real_images = real_images;
#                             final_fake_images = fake_images;
#                             final_noisy_images = current_noisy_tensor
#
#
#                     # Get original frame numpy:
#                     frame_original = np.transpose(final_real_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
#                     frame_original = np.uint8(frame_original.round())
#                     # Get noisy frame numpy:
#                     frame_noisy = np.transpose(final_noisy_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
#                     frame_noisy = np.uint8(frame_noisy.round())
#                     # Get clean frame numpy:
#                     frame_clean = np.transpose(final_fake_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
#                     frame_clean = np.uint8(frame_clean.round())
#                     # Get Concat frame numpy:
#                     frame_concat = np.concatenate((frame_original, frame_noisy, frame_clean), axis=1)
#
#                     noisy_min = fake_images.min();
#                     noisy_max = fake_images.max();
#
#                     ### Write Frame To Video: ###
#                     if flag_save_videos:
#                         if flag_show_image_or_highpass_details==0:
#                             pause(0.01);
#                             video_writer_Concat_ALL_DETAILS.write(frame_concat)
#                             pause(0.01);
#                             video_writer_original_ALL_DETAILS.write(frame_original);
#                             pause(0.01);
#                             video_writer_noisy_ALL_DETAILS.write(frame_noisy);
#                             pause(0.01);
#                             video_writer_clean_ALL_DETAILS.write(frame_clean);
#                         if flag_show_image_or_highpass_details==1:
#                             pause(0.01);
#                             video_writer_Concat_HIGHPASS.write(frame_concat)
#                             pause(0.01);
#                             video_writer_original_HIGHPASS.write(frame_original);
#                             pause(0.01);
#                             video_writer_noisy_HIGHPASS.write(frame_noisy);
#                             pause(0.01);
#                             video_writer_clean_HIGHPASS.write(frame_clean);
#
#                     ### Write Seperate Frames To Folders: ###
#                     if flag_save_video_or_images == 'images' or flag_save_video_or_images == 'both':
#                         if flag_show_image_or_highpass_details == 0:
#                             save_image_numpy(original_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_original, flag_convert_bgr2rgb=False, flag_scale=False)
#                             save_image_numpy(noisy_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
#                             save_image_numpy(clean_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_clean, flag_convert_bgr2rgb=False, flag_scale=False)
#                             save_image_numpy(Concat_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_concat, flag_convert_bgr2rgb=False, flag_scale=False)
#                         if flag_show_image_or_highpass_details == 1:
#                             save_image_numpy(original_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_original, flag_convert_bgr2rgb=False, flag_scale=False)
#                             save_image_numpy(noisy_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
#                             save_image_numpy(clean_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_clean, flag_convert_bgr2rgb=False, flag_scale=False)
#                             save_image_numpy(Concat_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_so_far).rjust(10, '0') + '.png', numpy_array=frame_concat, flag_convert_bgr2rgb=False, flag_scale=False)
#
#     ### END Of Generator Loss Accumulation! ###
#
#
#
#     ### Print To Console: ###
#     # Get Print String:
#     # number_of_same_image_steps += 1;
#     # number_of_time_steps_so_far += 1;
#     print_string = 'G Step ' + scientific_notation(number_of_time_steps_so_far) + ', '
#     if flag_use_validity_loss:
#         print_string += 'Val Loss: ' + scientific_notation(current_generator_validity_loss.item()) + ', '
#         print_string += 'Fake Val Acc: ' + str(float(current_accuracy_fake_images)) + ', '
#         print_string += 'Real Val Acc: ' + str(float(current_accuracy_real_images)) + ', '
#     if flag_use_direct_pixels_L1_loss:
#         print_string += 'Direct L1: ' + scientific_notation(current_generator_direct_pixels_loss.item()) + ', '
#     if flag_use_feature_extractor_direct_loss:
#         print_string += 'Features L1: ' + scientific_notation(current_generator_feature_extractor_loss.item()) + ', '
#     if flag_use_direct_pixels_contextual_loss:
#         print_string += 'Direct Contextual:' + scientific_notation(current_generator_direct_pixels_contextual_loss.item()) + ', '
#     if flag_use_feature_extractor_contextual_loss:
#         print_string += 'Features Contextual:' + scientific_notation(current_generator_feature_contextual_loss.item()) + ', '
#     if flag_use_gradient_sensitive_loss:
#         print_string += 'GradientSensitive Loss: ' + scientific_notation(current_generator_direct_pixels_gradient_sensitive_loss.item()) + ', '
#     if flag_use_direct_pixels_gram_loss:
#         print_string += 'Direct Gram: ' + scientific_notation(current_generator_direct_pixels_gram_loss.item()) + ', '
#     if flag_use_feature_extractor_gram_loss:
#         print_string += 'Features Gram: ' + scientific_notation(current_generator_feature_extractor_gram_loss.item()) + ', '
#     toc(print_string)
#
#
#     ####################################
#     ### Back-Propagate: ###
#     print(total_generator_loss)
#
#
#
# if flag_save_videos:
#     cv2.destroyAllWindows()
#     video_writer_original_ALL_DETAILS.release()
#     video_writer_noisy_ALL_DETAILS.release()
#     video_writer_clean_ALL_DETAILS.release()
#     video_writer_Concat_ALL_DETAILS.release()
#     video_writer_original_HIGHPASS.release()
#     video_writer_noisy_HIGHPASS.release()
#     video_writer_clean_HIGHPASS.release()
#     video_writer_Concat_HIGHPASS.release()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #############################################################################################################################################################################################################################
# ### Show Video Results: ###
# #(1). Start Recording Video:
# image_index = 1;
# created_video_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos/TNR1.avi'
# fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
#
#
# #(2). Get Video To Test On:
# test_video_full_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl/welcome back full movie hd.mp4'
# video_stream = cv2.VideoCapture(test_video_full_filename)
#
#
# #(3). Prepare Generator:
# Generator_Network = freeze_gradients(Generator_Network)
# Generator_Network.reset_hidden_states()
# Generator_Network = Generator_to_eval_mode(Generator_Network)
# Generator_device = device1
# Generator_Network = Generator_to_GPU(Generator_Network, netF, device1)
# noise_sigma = 0.1
# current_step = 0;
# current_valid_step = 0;
# current_image = 0;
# flag_limit_number_of_batches = True
# max_number_of_batches = 2;
# max_number_of_images = 50
# start_frame = 8*1000
#
#
#
# ### Show Video Results: ###
# #(1). Start Recording Video:
# image_index = 1;
# fps = 30.0
# flag_save_videos = True;
# flag_show_results = False;
# flag_save_video_or_images = 'videos' #'images' / 'videos'
# crop_size = 20000;
# created_video_filename_prefix = load_Generator_filename.split('.')[0]
# super_folder_to_save_result_videos = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos/'
# sub_folder_to_save_current_results_videos = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos/' + created_video_filename_prefix;
# if not os.path.exists(sub_folder_to_save_current_results_videos):
#     os.makedirs(sub_folder_to_save_current_results_videos)
# video_postfix_type = '.mpeg'
# created_video_filename_original = os.path.join(sub_folder_to_save_current_results_videos, 'original' + video_postfix_type)
# created_video_filename_noisy = os.path.join(sub_folder_to_save_current_results_videos, 'noisy' + video_postfix_type)
# created_video_filename_clean = os.path.join(sub_folder_to_save_current_results_videos, 'clean' + video_postfix_type)
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # Be sure to use lower case  MP42
#
# if flag_save_video_or_images == 'images':
#     flag_save_videos = False
#     original_video_images_folder = os.path.join(sub_folder_to_save_current_results_videos, 'original')
#     noisy_video_images_folder = os.path.join(sub_folder_to_save_current_results_videos, 'noisy')
#     clean_video_images_folder = os.path.join(sub_folder_to_save_current_results_videos, 'clean')
#     if not os.path.exists(original_video_images_folder):
#         os.makedirs(original_video_images_folder)
#     if not os.path.exists(noisy_video_images_folder):
#         os.makedirs(noisy_video_images_folder)
#     if not os.path.exists(clean_video_images_folder):
#         os.makedirs(clean_video_images_folder)
#
#
# #(2). Get Video To Test On:
# test_video_full_filename = 'F:\Movies\Raw Films/Night_videowalk_in_East_Shinjuku_Tokyo.mkv'
# video_stream = cv2.VideoCapture(test_video_full_filename)
#
#
# #(3). Prepare Generator:
# # Generator_Network = load_Generator_from_checkpoint(models_folder, load_Generator_filename, None);
#
# Generator_Network = freeze_gradients(Generator_Network)
# Generator_Network.reset_hidden_states()
# Generator_Network = Generator_to_eval_mode(Generator_Network)
# Generator_device = device0
# Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device)
# noise_sigma = 0.1
# current_step = 0;
# current_valid_step = 0;
# current_image = 0;
# flag_limit_number_of_batches = True
# max_number_of_batches = 2;
# max_number_of_images = 30*15
# start_frame = 1*100
# # flag_residual_learning = False;
#
# #(4.1). Test Using UnCropped Video:
# flag_frame_available = True;
# current_frame = train_dataset[0];   ###  Image to read!!!! ###
# while current_valid_step <= max_number_of_images and flag_frame_available:
#     tic()
#     current_step += 1;
#
#     if current_step == 1:
#         frame_height, frame_width, number_of_channels = current_frame.shape
#         crop_size_width = min(crop_size, frame_width);
#         crop_size_height = min(crop_size, frame_height);
#         start_index_height = random_integers(0,frame_height - crop_size_height);
#         start_index_width = random_integers(0,frame_width - crop_size_width)
#         if flag_save_videos:
#             video_writer_original = cv2.VideoWriter(created_video_filename_original, fourcc, fps, (frame_width*1, frame_height))
#             video_writer_noisy = cv2.VideoWriter(created_video_filename_noisy, fourcc, fps, (frame_width * 1, frame_height))
#             video_writer_clean = cv2.VideoWriter(created_video_filename_clean, fourcc, fps, (frame_width * 1, frame_height))
#
#     if current_step >= start_frame:
#         current_valid_step += 1;
#
#         #   (2). Get To Tensor:
#         current_frame_cropped = current_frame[start_index_height:start_index_height+crop_size_height, start_index_width:start_index_width+crop_size_width, :]
#         current_frame_transposed = np.expand_dims(np.transpose(current_frame_cropped, [2,0,1]), 0);
#         current_clean_tensor = torch.Tensor(current_frame_transposed) / 255;
#         current_clean_tensor = current_clean_tensor.to(Generator_device);
#
#         # (3). Add Noise:
#         dshape = current_clean_tensor.shape
#         dshape = np.array(dshape)
#         # current_noisy_tensor = current_noisy_tensor + torch.Tensor(randn(dshape[0], dshape[1], dshape[2], dshape[3])).to(Generator_device) * noise_sigma
#         current_noisy_tensor_multiplicative_noise = current_clean_tensor.mean(1, True) * noise_sigma * torch.Tensor(randn(dshape[0], dshape[1], dshape[2], dshape[3])).to(Generator_device)
#         current_noisy_tensor = current_clean_tensor + current_noisy_tensor_multiplicative_noise  # Multiplicative Noise
#
#         # (4). Pass noisy input through network:
#         # current_noisy_tensor = Variable(current_noisy_tensor, requires_grad=False)
#         output_tensor = Generator_Network(current_noisy_tensor);
#
#         if flag_residual_learning:
#             output_tensor += gaussian_blur_layer(current_noisy_tensor)
#
#         # (5). Get L1 between input and output:
#         L1_loss = nn.L1Loss()(current_clean_tensor, output_tensor).item()
#
#         ### Turn to uint8 numpy: ###
#         # Get original frame numpy:
#         frame_original = np.transpose(current_clean_tensor.clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
#         frame_original = np.uint8(frame_original.round())
#         # Get noisy frame numpy:
#         frame_noisy = np.transpose(current_noisy_tensor.clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
#         frame_noisy = np.uint8(frame_noisy.round())
#         # Get clean frame numpy:
#         frame_clean = np.transpose(output_tensor.clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
#         frame_clean = np.uint8(frame_clean.round())
#
#
#         ### Show Results: ###
#         if flag_show_results:
#             subplot(3, 1, 1)
#             imshow(frame_original)
#             title('original image')
#             subplot(3, 1, 2)
#             imshow(frame_noisy)
#             title('noisy image')
#             subplot(3, 1, 3)
#             imshow(frame_clean)
#             title('output image. ' + 'Current Step: ' + str(current_step) + ', L1 Loss: ' + str(L1_loss))
#             pause(0.01)
#
#         noisy_min = output_tensor.min();
#         noisy_max = output_tensor.max();
#
#
#         ### Write Frame To Video: ###
#         if flag_save_videos:
#             # Get Concatenated frame:
#             # frame_total = np.concatenate((frame_original, frame_noisy, frame_clean), axis=1)
#             pause(0.01);
#             video_writer_original.write(frame_original);
#             pause(0.01);
#             video_writer_noisy.write(frame_noisy);
#             pause(0.01);
#             video_writer_clean.write(frame_clean);
#
#         ### Write Seperate Frames To Folders: ###
#         if flag_save_video_or_images == 'images':
#             save_image_numpy(original_video_images_folder, filename=str(current_step).rjust(10,'0')+'.png', numpy_array=frame_original, flag_convert_bgr2rgb=False, flag_scale=False)
#             save_image_numpy(noisy_video_images_folder, filename=str(current_step).rjust(10, '0')+'.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
#             save_image_numpy(clean_video_images_folder, filename=str(current_step).rjust(10, '0')+'.png', numpy_array=frame_clean, flag_convert_bgr2rgb=False, flag_scale=False)
#
#         toc('frame no: ' + str(current_step) + ', L1 Loss: ' + str(L1_loss) + ', noisy_max: ' + str(noisy_max) + ', noisy_min: ' + str(noisy_min));
#
# if flag_save_videos:
#     cv2.destroyAllWindows()
#     video_writer_original.release()
#     video_writer_noisy.release()
#     video_writer_clean.release()
#
# ### Saved Video Is Not Pretty: ###
# created_video_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Example Videos/TNR1.avi'
# video_stream = cv2.VideoCapture(created_video_filename)
# flag_frame_available, frame = video_stream.read()
# imshow(frame)
# #############################################################################################################################################################################################################################
#
#
#
#
#
#








