
#TODO: add to RapidBase.Utils.Registration_Utils
def perform_gimbaless_cross_correlation_on_image_batch(input_tensor, number_of_samples_per_inner_batch):
    # ####################################################################################################
    # ### Testing: ###
    # shift_layer = Shift_Layer_Torch()
    # number_of_samples = 25
    # number_of_samples_per_inner_batch = 5
    # number_of_batches = number_of_samples // number_of_samples_per_inner_batch
    # SNR = 10
    # #(*). Temp - speckle pattern:
    # speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(10, 70,0,1,1,0)
    # speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    # #(*). Temp - real image:
    # speckle_pattern = read_image_default_torch()
    # speckle_pattern = RGB2BW(speckle_pattern)
    # speckle_pattern = crop_torch_batch(speckle_pattern, 512).squeeze(0)
    # C, H, W = speckle_pattern.shape
    # ### Get Input Tensor: ###
    # input_tensor = torch.cat([speckle_pattern] * number_of_samples, 0)
    # current_shift = 0
    # ### Actually Shift: ###
    # # (1). Batch Processing:
    # real_shifts_vec = torch.randn(number_of_samples)
    # differential_shifts_vec = real_shifts_vec[1:number_of_samples] - real_shifts_vec[0:number_of_samples-1]
    # differential_shifts_vec = differential_shifts_vec * -1
    # real_shifts_to_center_frame = real_shifts_vec - real_shifts_vec[number_of_samples//2]
    # real_shifts_to_center_frame = torch.cat([real_shifts_to_center_frame[0:number_of_samples//2], real_shifts_to_center_frame[number_of_samples//2+1:]])
    # input_tensor = shift_layer.forward(input_tensor, real_shifts_vec.cpu().numpy(), real_shifts_vec.cpu().numpy())
    # input_tensor = crop_torch_batch(input_tensor, 512)
    # input_tensor = input_tensor.unsqueeze(0)
    # #

    ### Add Noise To Tensor: ###
    B,C,H,W = input_tensor.shape
    number_of_samples = C
    number_of_batches = number_of_samples // number_of_samples_per_inner_batch
    final_outputs = []
    for i in np.arange(number_of_batches):
        start_index = i*number_of_samples_per_inner_batch
        stop_index = start_index + number_of_samples_per_inner_batch
        current_mean_frame = align_image_batch_to_center_frame_cross_correlation(input_tensor[:,start_index:stop_index])
        final_outputs.append(current_mean_frame)
    final_outputs = torch.cat(final_outputs,1)

    # imshow_torch(input_tensor[:,12])
    # imshow_torch(final_outputs[0])

    return final_outputs

def align_image_batch_to_center_frame_cross_correlation(input_tensor):
    # ####################################################################################################
    # ### Testing: ###
    shift_layer = Shift_Layer_Torch()
    # number_of_samples = 11
    # SNR = 10
    # speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(10, 70,0,1,1,0)
    # speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    # #(*). Temp - real image:
    # speckle_pattern = read_image_default_torch()
    # speckle_pattern = RGB2BW(speckle_pattern)
    # speckle_pattern = crop_torch_batch(speckle_pattern, 512).squeeze(0)
    # C, H, W = speckle_pattern.shape
    # ### Get Input Tensor: ###
    # input_tensor = torch.cat([speckle_pattern] * number_of_samples, 0)
    # current_shift = 0
    # ### Actually Shift: ###
    # # (1). Batch Processing:
    # real_shifts_vec = torch.randn(number_of_samples)
    # differential_shifts_vec = real_shifts_vec[1:number_of_samples] - real_shifts_vec[0:number_of_samples-1]
    # differential_shifts_vec = differential_shifts_vec * -1
    # real_shifts_to_center_frame = real_shifts_vec - real_shifts_vec[number_of_samples//2]
    # real_shifts_to_center_frame = torch.cat([real_shifts_to_center_frame[0:number_of_samples//2], real_shifts_to_center_frame[number_of_samples//2+1:]])
    # input_tensor = shift_layer.forward(input_tensor, real_shifts_vec.cpu().numpy(), real_shifts_vec.cpu().numpy())
    # input_tensor = crop_torch_batch(input_tensor, 512)
    # input_tensor = input_tensor.unsqueeze(0)

    # ### Add Noise To Tensor: ###
    # B,C,H,W = input_tensor.shape
    # number_of_samples = B
    # noise_map = torch.randn_like(input_tensor) * 1/np.sqrt(SNR)
    # input_tensor += noise_map
    ####################################################################################################

    B, C, H, W = input_tensor.shape
    number_of_samples = B

    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:,2] + y[:,0] - 2 * y[:,1]) / 2
        b = -(y[:,0] + 2 * a * x[1] - y[:,1] - a)
        c = y[:,1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    #########################################################################################################
    ### Get CC: ###
    #input tensor [T,H,W]
    B,T,H,W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1,-2])
    #(1). Regular Cross Corerlation:
    output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * input_tensor_fft[:, T//2, :, :].conj(), dim=[-1, -2]).real
    # output_CC = torch.cat()
    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H*W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    #(1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    #(2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0*W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1*W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1*W
    output_CC_flattened_indices_i0 = i1 + i0*W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H*W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    shifted_tensors = shift_layer.forward(input_tensor.permute([1,0,2,3]), -shiftx, -shifty) #TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0,True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return mean_frame_averaged



#TODO: add to pre_post_processing
class PreProcessing_EDVR_Gimbaless(PreProcessing_Denoise_NonRecursive_Base):
    def __init__(self, **kw):
        super(PreProcessing_EDVR_Gimbaless, self).__init__()
        # self.flag_BW2RGB = True
        self.flag_BW2RGB = False

    def forward(self, inputs_dict, Train_dict, model):
        ### Names Reassignment: ###
        id = inputs_dict
        td = Train_dict

        ### Adjust Preprocessing Parameters: ###
        network_input, id, td, model = self.prepare_inputs_for_net(id, td, model)
        id, td, model = self.adjust_inputs_dict_for_loss(id, td, model)

        ### Gimbaless - align and avearge images: ###
        output_tensors_list = []
        number_of_samples_per_sub_batch = 25
        for i in np.arange(id.output_frames_noisy.shape[0]):
            output_tensors_list.append(perform_gimbaless_cross_correlation_on_image_batch(id.output_frames_noisy[i].permute([1,0,2,3]), number_of_samples_per_sub_batch).unsqueeze(0))
        output_tensors = torch.cat(output_tensors_list)


        # imshow_torch(output_tensors[0,0,0]);
        # imshow_torch(output_tensors[0,0,1]);
        # imshow_torch(id.output_frames_noisy[0,0]);
        # imshow_torch(id.output_frames_noisy[0,1]);
        # imshow_torch(id.output_frames_original[0, 12]);
        # imshow_torch(id.output_frames_original[0, 1]);
        return network_input, id, td, model



