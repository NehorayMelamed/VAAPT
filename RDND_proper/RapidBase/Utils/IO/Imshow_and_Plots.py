import cv2

import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

# ### Other Utils: ###
# import RapidBase.Utils.Pytorch_Utils
# import RapidBase.Utils.Adding_Noise_To_Image
# import RapidBase.Utils.GPU_Utils
# # import RapidBase.Utils.Imshow_and_Plots
# import RapidBase.Utils.Klepto
# import RapidBase.Utils.linspace_arange
# import RapidBase.Utils.MISCELENEOUS
# import RapidBase.Utils.Path_and_Reading_utils
# import RapidBase.Utils.tic_toc
# from RapidBase.Utils.Pytorch_Utils import *
# from RapidBase.Utils.Adding_Noise_To_Image import *
# from RapidBase.Utils.IO.GPU_Utils import *
# # from RapidBase.Utils.IO.Imshow_and_Plots import *
# from RapidBase.Utils.IO.Klepto import *
# from RapidBase.Utils.Tensor_Manipulation.linspace_arange import *
# from RapidBase.Utils.MISCELENEOUS import *
# from RapidBase.Utils.IO.Path_and_Reading_utils import *
# from RapidBase.Utils.IO.tic_toc import *
from RapidBase.Utils.Classical_DSP.Convolution_Utils import convn_layer_torch


def imshow2(image):
    pylab.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))



def save_image_torch(folder_path=None, filename=None, torch_tensor=None, flag_convert_bgr2rgb=True, flag_scale_by_255=False, flag_array_to_uint8=True, flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False, scale_ref_tensor=None, mask_tensor = None):
    if flag_scale_by_255:
        scale_factor = 255
    else:
        scale_factor = 1

    if flag_convert_bgr2rgb:
        saved_array = cv2.cvtColor(torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor, cv2.COLOR_BGR2RGB)
    else:
        saved_array = torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor

    if(mask_tensor is None):
        mask_array = torch.ones_like(torch_tensor).data.cpu().numpy().transpose([1, 2, 0])
    else:
        mask_array = mask_tensor.data.cpu().numpy().transpose([1, 2, 0])

    if(mask_array.shape[2]==1):
        mask_array = mask_array.repeat(3, 2)

    if(scale_ref_tensor is None or scale_ref_tensor.min()==scale_ref_tensor.max()):
        min_val = saved_array.min()
        max_val = saved_array.max()
    else:
        if flag_convert_bgr2rgb:
            saved_array_ref = cv2.cvtColor(scale_ref_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor,
                                       cv2.COLOR_BGR2RGB)
        else:
            saved_array_ref = scale_ref_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale_factor

        min_val = saved_array.min()
        max_val = saved_array.max()

        min_val_ref = saved_array_ref.min()
        max_val_ref = saved_array_ref.max()

        min_val = max(min_val, min_val_ref)
        max_val = min(max_val, max_val_ref)
        if max_val < min_val:  #if i don't have this condition it causes problems and i don't have the strength to think of anything smart
            min_val = min(min_val, min_val_ref)
            max_val = max(max_val, max_val_ref)


    if flag_convert_grayscale_to_heatmap:
        if torch_tensor.shape[0]==1:
            #(1). Direct Formula ColorMap:
            #saved_array = gray2color_numpy(saved_array,0)
            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=min_val, vmax=max_val)
            saved_array = 255 * cmap(norm(saved_array.squeeze()))
            saved_array = saved_array[:,:,0:3]
            #plt.imshow(saved_array)
            #plt.show()
            # saved_array = gray2color_new(saved_array)
            # #(2). Matplotlib Jet:
            # cmap = plt.cm.jet
            # norm = plt.Normalize(vmin=0, vmax=150)
            # gt_disparity = saved_array
            # gt_disparity2 = norm(gt_disparity)
            # gt_disparity3 = cmap(gt_disparity2)
            # saved_array = 255 * gt_disparity3[:,:,0:3]


    # path_make_path_if_none_exists(folder_path)

    if flag_imagesc:
        new_range = (0, 255)
        new_range_delta = new_range[1]-new_range[0]
        old_range_delta = saved_array.max() - saved_array.min()
        new_min = new_range[0]
        old_min = saved_array.min()
        saved_array = ((saved_array-old_min)*new_range_delta/old_range_delta) + new_min

    saved_array = saved_array * mask_array

    if flag_array_to_uint8:
        saved_array = saved_array.astype('uint8')

    if flag_convert_grayscale_to_heatmap:
        saved_array = cv2.cvtColor(saved_array, cv2.COLOR_BGR2RGB)

    if flag_save_figure:
        if len(saved_array.shape)==3:
            if saved_array.shape[2] == 1:
                imshow(saved_array.squeeze())
            else:
                imshow(saved_array)
        else:
            imshow(saved_array)
        plt.savefig(os.path.join(folder_path, filename))
    else:
        cv2.imwrite(os.path.join(folder_path, filename), saved_array)

    print(os.path.join(folder_path, filename))



import matplotlib.pylab as pylab
from matplotlib.pylab import figure, colorbar, title
def stack_image_BW2RGB(input_image):
    if len(input_image.squeeze().shape)==2:
        input_image = np.atleast_3d(input_image)
        input_image = np.concatenate([input_image, input_image, input_image],-1)
        return input_image
    else:
        return input_image

def imshow_torch(image, flag_colorbar=True, title_str='', flag_maximize=False):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times
    title(title_str)
    plt.show()

    if flag_maximize:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    return fig


from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import scale_array_to_range, scale_array_stretch_hist, scale_array_from_range
def imagesc_torch(image, flag_colorbar=True, title_str='', flag_maximize=False):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 3 or len(image.shape) == 4:
        image_final = image[0]
    else:
        image_final = image
    image_final = scale_array_to_range(image_final)

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image_final.detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image_final.detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image_final.detach().cpu().numpy())
    if flag_colorbar:
        colorbar()
    title(title_str)

    if flag_maximize:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    return fig

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_histogram_equalize, scale_array_clahe
def imagesc_hist_torch(image, flag_colorbar=True, title_str='', flag_maximize=False):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 3 or len(image.shape) == 4:
        image_final = image[0]
    else:
        image_final = image
    image_final = scale_array_stretch_hist(image_final)

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image_final.detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image_final.detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image_final.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()
    title(title_str)

    if flag_maximize:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    return fig


def imagesc_clahe_torch(image, flag_colorbar=True, title_str='', flag_maximize=False):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 5:
        image_final = image[0,0].unsqueeze(0)
    elif len(image.shape) == 4:
        image_final = image[0].unsqueeze(0)
    elif len(image.shape) == 3:
        image_final = image.unsqueeze(0)
    elif len(image.shape) == 2:
        image_final = image.unsqueeze(0).unsqueeze(0)
    image_final = scale_array_clahe(image_final, flag_stretch_histogram_first=True, quantiles=(0.01,0.99))
    image_final = image_final[0]
    pylab.imshow(np.transpose(image_final.detach().cpu().numpy(), (1, 2, 0)).squeeze())

    if flag_colorbar:
        colorbar()
    title(title_str)

    if flag_maximize:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    return fig

def imshow_torch_histogram_stretch(image, flag_colorbar=True, title_str='', flag_maximize=False):
    imagesc_hist_torch(image, flag_colorbar, title_str, flag_maximize)

def imshow_torch_clahe(image, flag_colorbar=True, title_str='', flag_maximize=False):
    imagesc_clahe_torch(image, flag_colorbar, title_str, flag_maximize)

def imshow_torch_imagesc(image, flag_colorbar=True, title_str='', flag_maximize=False):
    imagesc_torch(image, flag_colorbar, title_str, flag_maximize)

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import get_full_shape_torch
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import BW2RGB, torch_to_numpy
def imshow_torch_video(input_tensor, number_of_frames=None, FPS=3, flag_BGR2RGB=True, frame_stride=1, flag_colorbar=False, video_title='', video_title_list=None):
    #TODO: fix colorbar
    def get_correct_form(input_tensor, i):
        if shape_len == 4:
            #(T,C,H,W)
            if flag_BGR2RGB and C==3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 3:
            #(T,H,W)
            output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 5:
            #(B,T,C,H,W)
            if flag_BGR2RGB and C==3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[0, i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[0, i])
        return output_tensor

    ### Get Parameters: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if number_of_frames is not None:
        number_of_frames_to_show = min(number_of_frames, T)
    else:
        number_of_frames_to_show = T

    cf = figure()
    # plt.ion() #TODO: what's this?
    output_tensor = get_correct_form(input_tensor, 0)
    im = plt.imshow(output_tensor)
    if flag_colorbar:
        cbar = plt.colorbar(im)
    for i in np.arange(0,number_of_frames_to_show,frame_stride):
        output_tensor = get_correct_form(input_tensor, i)
        im.set_array(output_tensor)
        plt.show()
        if video_title_list is not None:
            current_title = str(video_title_list[i])
        else:
            current_title = video_title
        plt.title(current_title + '  ' + str(i))
        plt.pause(1/FPS)
        if flag_colorbar:
            cbar = plt.colorbar(im)
            plt.draw()
    plt.ioff()


def imshow_torch_video_running_mean(input_tensor, number_of_frames=None, FPS=3, running_mean_size=3, flag_BGR2RGB=True, frame_stride=1):

    def get_correct_form(input_tensor, i):
        if shape_len == 4:
            #(T,C,H,W)
            if flag_BGR2RGB and C==3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 3:
            #(T,H,W)
            output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 5:
            #(B,T,C,H,W)
            if flag_BGR2RGB and C==3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[0, i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[0, i])
        return output_tensor

    ### Get Parameters: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if number_of_frames is not None:
        number_of_frames_to_show = min(number_of_frames, T)
    else:
        number_of_frames_to_show = T

    ### Get running mean over time: ###
    convn_layer = convn_layer_torch()
    output_tensor = convn_layer.forward(input_tensor, torch.ones(running_mean_size).to(input_tensor.device), 0)
    # output_tensor = convn_layer.forward(input_tensor, np.ones(running_mean_size), 0)

    cf = figure()
    # plt.ion() #TODO: what's this?
    current_output_tensor = get_correct_form(output_tensor, 0)
    im = plt.imshow(current_output_tensor)
    for i in np.arange(0,number_of_frames_to_show,frame_stride):
        current_output_tensor = get_correct_form(output_tensor, i)
        im.set_array(current_output_tensor)
        plt.show()
        plt.title(str(i))
        plt.pause(1/FPS)
    plt.ioff()


def imshow_numpy_list_video(input_list, number_of_frames=None, FPS=3, flag_BGR2RGB=True, frame_stride=1):

    # def get_correct_form(input_tensor, i):
    #     if shape_len == 4:
    #         #(T,C,H,W)
    #         if flag_BGR2RGB and C==3:
    #             output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[i]), cv2.COLOR_BGR2RGB)
    #         else:
    #             output_tensor = torch_to_numpy(input_tensor[i])
    #     elif shape_len == 3:
    #         #(T,H,W)
    #         output_tensor = torch_to_numpy(input_tensor[i])
    #     elif shape_len == 5:
    #         #(B,T,C,H,W)
    #         if flag_BGR2RGB and C==3:
    #             output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[0, i]), cv2.COLOR_BGR2RGB)
    #         else:
    #             output_tensor = torch_to_numpy(input_tensor[0, i])
    #     return output_tensor

    ### Get Parameters: ###
    T = len(input_list)
    if number_of_frames is not None:
        number_of_frames_to_show = min(number_of_frames, T)
    else:
        number_of_frames_to_show = T

    cf = figure()
    # plt.ion() #TODO: what's this?
    # output_tensor = get_correct_form(input_list, 0)
    output_tensor = input_list[0]
    im = plt.imshow(output_tensor)
    plt.colorbar()
    for i in np.arange(0,number_of_frames_to_show,frame_stride):
        # output_tensor = get_correct_form(input_list, i)
        output_tensor = input_list[i]
        im.set_array(output_tensor)
        plt.show()
        plt.title(str(i))
        plt.pause(1/FPS)
    plt.ioff()

from matplotlib.animation import FuncAnimation
def fast_imshow():
    im = plt.imshow(np.random.randn(10,10))
    def update(i):
        A = np.random.randn(10, 10)
        im.set_array(A)
        return im
    ani = FuncAnimation(plt.gcf(), update, frames=range(100), interval=5, blit=False )  #maybe blit=True will be faster
    plt.show()


def imshow_torch_BW(image, flag_colorbar=True, title_str=''):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 4:
        pylab.imshow(stack_image_BW2RGB(np.transpose(image[0].detach().cpu().numpy(),
                                  (1, 2, 0)).squeeze()))  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(stack_image_BW2RGB(np.transpose(image.detach().cpu().numpy(),
                                  (1, 2, 0)).squeeze()))# transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()  # TODO: fix the bug of having multiple colorbars when calling this multiple times
    title(title_str)
    return fig


def imshow_torch_multiple(images_tuple, flag_colorbar=True, title_str=''):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    image = images_tuple[0]
    for i in np.arange(1,len(images_tuple)):
        image = torch.cat([image,images_tuple[i]], -1)

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times
    title(title_str)
    return fig

def imshow_new(input_image):
    figure()
    mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    imshow(input_image)
    colorbar()

def plot_torch(input_signal1, input_signal2=None):
    if input_signal2 is None:
        # plt.cla()
        # plt.clf()
        # plt.close()
        plt.plot(np.arange(len(input_signal1.detach().cpu().numpy().squeeze())), input_signal1.detach().cpu().numpy().squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    else:
        # plt.clf()
        # plt.close()
        plt.plot(input_signal1.detach().cpu().numpy().squeeze(),input_signal2.detach().cpu().numpy().squeeze())






############################################################################################################################################

def plot_fft_graphs(input_tensor, H_center=None, W_center=None, area_around_center=None, specific_case_string='', params=None):
    # input_tensor = Movie_BGS[t_vec.type(torch.LongTensor)]
    # input_tensor = Movie_BGS
    # input_tensor = Movie[t_vec.type(torch.LongTensor)]
    # input_tensor = TrjMov.unsqueeze(1)
    # H_center = 279
    # W_center = 251
    # area_around_center = 3
    # specific_case_string = 'TrjMovie_BGS_1_pixel_right_constant'

    # ### Start from a certain frame (mostly for testing purposes if i don't want to start from the first frame): ###
    # frame_to_start_from = 500 * 5
    # f = initialize_binary_file_reader(f)
    # Movie_temp = read_frames_from_binary_file_stream(f, 1000, frame_to_start_from * 1, params)
    # Movie_temp = Movie_temp.astype(float)
    # # Movie_temp = scale_array_from_range(Movie_temp.clip(q1, q2),
    # #                                        min_max_values_to_clip=(q1, q2),
    # #                                        min_max_values_to_scale_to=(0, 1))
    # # Movie_temp = scale_array_stretch_hist(Movie_temp)
    # input_tensor = torch.tensor(Movie_temp).unsqueeze(1)

    # ### Read Certain Area: ###
    # frame_to_start_from = 500 * 5
    # f = initialize_binary_file_reader(f)
    # Movie_temp = read_frames_from_binary_file_stream_SpecificArea(f,
    #                                                                       number_of_frames_to_read=1010,
    #                                                                       number_of_frames_to_skip=frame_to_start_from * 1,
    #                                                                       params=params,
    #                                                                       center_HW=(279,251),
    #                                                                       area_around_center=10)
    # # Movie_temp = scale_array_from_range(Movie_temp.clip(q1, q2),
    # #                                        min_max_values_to_clip=(q1, q2),
    # #                                        min_max_values_to_scale_to=(0, 1))
    # # Movie_temp = scale_array_stretch_hist(Movie_temp)
    # input_tensor = torch.tensor(Movie_temp).unsqueeze(1)
    # imshow_torch_video(input_tensor, FPS=50, frame_stride=5)

    ### Get Drone Area: ###
    # Drone = [279, 251], Edge = [302, 221]
    if H_center is not None:
        i_vec = np.linspace(H_center - area_around_center, H_center + area_around_center, area_around_center * 2 + 1).astype(int)
        j_vec = np.linspace(W_center - area_around_center, W_center + area_around_center, area_around_center * 2 + 1).astype(int)
    else:
        i_vec = np.arange(input_tensor.shape[-2])
        j_vec = np.arange(input_tensor.shape[-1])

    ### Get FFT Over Entire Image: ###
    # imshow_torch_video(input_tensor, FPS=50)
    # input_tensor_fft = (torch.fft.rfftn(input_tensor, dim=0).abs())
    # input_tensor_fft = fftshift_torch(torch.fft.fftn(input_tensor, dim=0).abs(),0)
    # T,C,H,W = input_tensor_fft.shape

    ### Get FFT Over Specific Drone Area: ###
    input_tensor_drone = input_tensor[:, :, i_vec[0]:i_vec[-1] + 1, j_vec[0]:j_vec[-1] + 1].cuda()
    input_tensor_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_drone, dim=0).abs(), 0)
    T = input_tensor_fft.shape[0]

    ### Create Proper Folders: ###
    ffts_folder = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New')
    ffts_folder_before_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'Before_Conditions')
    ffts_folder_after_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'After_Conditions')
    path_create_path_if_none_exists(ffts_folder)
    path_create_path_if_none_exists(ffts_folder_before_conditions)
    path_create_path_if_none_exists(ffts_folder_after_conditions)

    ### Loop Over Individual Indices: ###
    final_frequency_peaks_list = []
    for i in np.arange(len(i_vec)):
        for j in np.arange(len(j_vec)):
            ### Initialize Tensor: ###
            # Drone = [279, 251], Edge = [302, 221]
            i = int(i)
            j = int(j)
            # input_tensor_fft_graph = input_tensor_fft[:, 0, int(i), int(j)]
            # input_tensor_fft_graph = convn_torch(input_tensor_fft_graph, torch.ones(3)/3, dim=0).squeeze()
            # input_tensor_fft_graph = input_tensor_fft_graph.clamp(0, 1)
            FPS = params.FPS
            frequency_axis = torch.tensor(FPS * np.linspace(-0.5, .5 - 1 / T, T))
            frequency_axis_numpy = frequency_axis.cpu().numpy()

            ### Initialize graphs location and string: ###
            graph_string = specific_case_string + '   ' + 'H=' + str(int(i_vec[i])) + '_W=' + str(int(j_vec[j]))

            ### Get Current 1D FFT-Vec: ###
            input_vec = input_tensor_fft[:, 0:1, int(i):int(i + 1), int(j):int(j + 1)].abs().cpu()
            # input_vec_save_full_filename = os.path.join(ffts_folder_before_conditions, graph_string + '.npy')
            # np.save(input_vec_save_full_filename, input_vec.cpu().numpy(), allow_pickle=True)

            ### Peak Detect: ###
            # input_vec = convn_torch(input_vec, torch.ones(3)/3, dim=0)
            maxima_peaks, arange_in_correct_dim_tensor = peak_detect_pytorch(input_vec,
                                                                             window_size=21,
                                                                             dim=0,
                                                                             flag_use_ratio_threshold=True,
                                                                             ratio_threshold=3,
                                                                             flag_plot=True)
            # (*). Only Keep Peaks Above Median Noise-Floor Enough:
            median_noise_floor = input_vec.median().item()
            SNR_threshold = median_noise_floor * 0
            logical_mask_above_noise_median = input_vec > SNR_threshold
            maxima_peaks = maxima_peaks * logical_mask_above_noise_median

            ######### Plot Graphs Between Harmonics Conditions: #########
            ### Get All Peak Frequencies: ###
            maxima_peaks_to_plot = (arange_in_correct_dim_tensor)[maxima_peaks]
            maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_to_plot] > 0
            peak_frequencies_list = frequency_axis[maxima_peaks_to_plot][maxima_frequency_peaks_logical_mask]
            peak_frequencies_list = peak_frequencies_list.tolist()
            peak_frequencies_array = np.array(peak_frequencies_list)
            ### Legends list: ###
            legends_list = copy.deepcopy(peak_frequencies_list)
            for legend_index in np.arange(len(peak_frequencies_list)):
                legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
            legends_list = ['FFT', 'peaks', 'Noise Floor', 'Threshold'] + legends_list
            ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
            k1 = 0
            k2 = 0
            k3 = 0
            maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
            plt.figure()
            plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0, 1).numpy())
            # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
            plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0, 1).numpy(), '.')
            plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * median_noise_floor)
            plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold)
            plt.title(graph_string)
            for current_legend in legends_list:
                plt.plot(frequency_axis_numpy, frequency_axis_numpy * 0)
            plt.legend(legends_list)
            ### Set Y limits: ###
            plt.ylim([0,1])
            ### Save Graph: ###
            save_string = os.path.join(ffts_folder_before_conditions, graph_string + '.png')
            plt.savefig(save_string)
            plt.close('all')

            # ########### Plot Graphs After Harmonics Conditions: ###############
            # # (*). Get Rid Of Peaks Which Are Harmonies Of The Base 23[Hz]:
            # base_frequency = 23
            # base_frequency_harmonic_tolerance = 1.5
            # frequency_axis_remainder_from_base_frequency = torch.remainder(frequency_axis, base_frequency)
            # # Instead of creating an array of [23,23,23.....,46,46,46,....] and getting the diff i simply check by a different, maybe stupid way for close harmonics
            # frequency_axis_modulo_base_logical_mask = frequency_axis_remainder_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from above
            # frequency_axis_diff_from_base_frequency = (frequency_axis_remainder_from_base_frequency - base_frequency).abs()
            # frequency_axis_modulo_base_logical_mask *= frequency_axis_diff_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from below
            # frequency_axis_modulo_base_logical_mask = frequency_axis_modulo_base_logical_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # maxima_peaks = maxima_peaks * frequency_axis_modulo_base_logical_mask
            # # (*). Get Rid Of Negative Frequencies:
            # maxima_peaks[0:T // 2] = False
            #
            # ### Get All Peak Frequencies: ###
            # maxima_peaks_to_plot = (arange_in_correct_dim_tensor)[maxima_peaks]
            # maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_to_plot] > 0
            # peak_frequencies_list = frequency_axis[maxima_peaks_to_plot][maxima_frequency_peaks_logical_mask]
            # peak_frequencies_list = peak_frequencies_list.tolist()
            # peak_frequencies_array = np.array(peak_frequencies_list)
            # ### Record Peak Frequencies: ###
            # for peak_frequency in peak_frequencies_list:
            #     final_frequency_peaks_list.append(round_to_nearest_half(peak_frequency))
            # ### Legends list: ###
            # legends_list = copy.deepcopy(peak_frequencies_list)
            # for legend_index in np.arange(len(peak_frequencies_list)):
            #     legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
            # legends_list = ['FFT', 'peaks', 'Noise Floor', 'Threshold'] + legends_list
            # ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
            # k1 = 0
            # k2 = 0
            # k3 = 0
            # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
            # plt.figure()
            # plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0, 1).numpy())
            # # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
            # plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0, 1).numpy(), '.')
            # plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * median_noise_floor)
            # plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold)
            # plt.title(graph_string)
            # for current_legend in legends_list:
            #     plt.plot(frequency_axis_numpy, frequency_axis_numpy * 0)
            # plt.legend(legends_list)
            # ### Save Graph: ###
            # save_string = os.path.join(ffts_folder_after_conditions, graph_string + '.png')
            # plt.savefig(save_string)
            # plt.close('all')

            # ### Save Array: ###
            # np.save(os.path.join(ffts_folder, graph_string + '.npy'), input_tensor_fft_graph.cpu().numpy())

            # ### Save Graph: ###
            # plt.figure()
            # plot_torch(frequency_axis, input_tensor_fft_graph)
            # plt.title(graph_string)
            # plt.show()
            # save_string = os.path.join(ffts_folder, graph_string + '.png')
            # plt.savefig(save_string)
            # plt.close('all')
    plt.close('all')


def plot_fft_graphs_original(flag_was_drone_found, params, trajectory_index,
                             input_tensor_fft,
                             input_tensor_fft_max_lobe_SNR_around_drone_bins,
                             input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough,
                             input_tensor_fft_noise_estimation_mean,
                             frequency_axis,
                             input_tensor_fft_possible_drone_bine_max_lobe_value,
                             drone_frequencies_axis,
                             noise_frequency_axis):
    flag_save = (flag_was_drone_found == True and params.was_there_drone == 'False')
    flag_save = flag_save or (flag_was_drone_found == False and params.was_there_drone == 'True')
    if params.flag_save_interim_graphs_and_movies:
        # TODO: add condition that if sum of low frequencies is very large, or that we can fit a lorenzian/gaussian on the low frequencies then it's probably a bird
        # (*). All Graphs:
        print('DRONE FOUND, SAVING FFT GRAPHS')
        flag_drone_found_but_no_drone_there = (flag_was_drone_found == True and params.was_there_drone == 'False')
        flag_drone_found_and_is_there = (flag_was_drone_found == True and params.was_there_drone == 'True')
        flag_drone_not_found_but_is_there = (flag_was_drone_found == False and params.was_there_drone == 'True')
        flag_drone_not_found_and_is_not_there = (flag_was_drone_found == False and params.was_there_drone == 'False')
        if flag_drone_not_found_but_is_there:
            post_string = '_Drone_Not_Found_But_Is_There(FT)'
        elif flag_drone_found_but_no_drone_there:
            post_string = '_Drone_Found_But_Is_Not_There(TF)'
        elif flag_drone_found_and_is_there:
            post_string = '_Drone_Found_And_Is_There(TT)'
        elif flag_drone_not_found_and_is_not_there:
            post_string = '_Drone_Not_Found_And_Is_Not_There(FF)'
        else:
            post_string = ''
        post_string = ''
        graphs_folder_unclipped = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_Unclipped')
        graphs_folder_clipped = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_clipped')
        graphs_folder_fft_binned = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_Binned_Clipped')
        path_make_path_if_none_exists(graphs_folder_fft_binned)
        path_make_path_if_none_exists(graphs_folder_unclipped)
        path_make_path_if_none_exists(graphs_folder_clipped)
        fft_counter = 0
        ROI_H, ROI_W = input_tensor_fft_max_lobe_SNR_around_drone_bins.shape[-2:]
        for roi_H in np.arange(ROI_H):
            for roi_W in np.arange(ROI_W):
                current_fft = input_tensor_fft[:, roi_H, roi_W].abs()
                current_SNR = input_tensor_fft_max_lobe_SNR_around_drone_bins[0, roi_H, roi_W].item()
                current_decision = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough[0, roi_H, roi_W].item()

                # ### Unclipped Graph: ##
                # figure()
                # plt.plot(frequency_axis, current_fft.abs().squeeze().cpu().numpy())
                # plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(drone_frequencies_axis, (+0.003) * np.ones_like(drone_frequencies_axis))
                # plt.plot(noise_frequency_axis, (+0.003 * 2) * np.ones_like(noise_frequency_axis))
                # plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
                # title_string = str('ROI = [' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(np.round(current_SNR * 10) / 10) + ', Decision = ' + str(current_decision))
                # file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
                # plt.title(title_string)
                # filename = os.path.join(graphs_folder_unclipped, file_string + '.png')
                # plt.savefig(filename)
                # plt.close()

                ### clipped Graph: ##
                figure()
                plt.plot(frequency_axis, current_fft.abs().squeeze().cpu().numpy().clip(0, 10))
                plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                plt.plot(drone_frequencies_axis, (+0.003) * np.ones_like(drone_frequencies_axis))
                plt.plot(noise_frequency_axis, (+0.003 * 2) * np.ones_like(noise_frequency_axis))
                plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
                title_string = str('[' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(current_SNR) + ', Decision = ' + str(current_decision))
                file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
                plt.title(title_string)
                ### Set Y limits: ###
                # plt.ylim([0, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * 3])
                plt.ylim([0, 1])
                filename = os.path.join(graphs_folder_clipped, file_string + '.png')
                plt.savefig(filename)
                plt.close()

                # ### Binned clipped Graph: ##
                # figure()
                # plt.plot(frequency_axis, input_tensor_fft_binned[:,roi_H, roi_W].abs().squeeze().cpu().numpy().clip(0, 10))
                # plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(drone_frequencies_axis, 0 * np.ones_like(drone_frequencies_axis))
                # plt.plot(noise_frequency_axis, (-0.003) * np.ones_like(noise_frequency_axis))
                # plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
                # title_string = str('[' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(current_SNR) + ', Decision = ' + str(current_decision))
                # file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
                # plt.title(title_string)
                # plt.ylim([0, 1])
                # filename = os.path.join(graphs_folder_clipped, file_string + '.png')
                # plt.savefig(filename)
                # plt.close()

                fft_counter += 1

                plt.close('all')
                plt.pause(0.1)

############################################################################################################################################



#############################################################################################################################################
def draw_polygon_points_on_image(image_frame, polygon_points):
    for i in np.arange(len(polygon_points)-1):
        if type(polygon_points[i]) != tuple:
            cv2.line(image_frame, tuple(polygon_points[i]), tuple(polygon_points[i + 1]), (255, 255, 255), 2)
        else:
            cv2.line(image_frame, polygon_points[i], polygon_points[i + 1], (255, 255, 255), 2)

    if type(polygon_points[-1]) != tuple:
        cv2.line(image_frame, tuple(polygon_points[-1]), tuple(polygon_points[0]), (255,255,255), 2)  #close the polygon
    else:
        cv2.line(image_frame, polygon_points[-1], polygon_points[0], (255, 255, 255), 2)

    return image_frame


def draw_polygons_on_image(image_frame, polygon_points):
    if polygon_points is None:
        return image_frame
    for i in np.arange(len(polygon_points)):
        image_frame = draw_polygon_points_on_image(image_frame, polygon_points[i])
    return image_frame


def draw_circles_with_labels_on_image(image_frame, circle_points, circle_radius_in_pixels=5, base_string='Trj', string_list=[]):
    #circle_points = a list of tuples/list/array  OR  a 2D-array of size [N,2].   each circle_point is (X0,Y0) of the circle
    #draws several circles and EACH CIRCLE WITH A DIFFERENT LABEL
    #TODO: add the possibility of inserting a list of circle_radius_in_pixels for different sized circles
    for i in np.arange(len(circle_points)):
        ### Get circle properties: ###
        current_circle_point = circle_points[i]
        if type(circle_radius_in_pixels) == list:
            current_circle_radius = circle_radius_in_pixels[i]
        else:
            current_circle_radius = circle_radius_in_pixels
        if string_list == []:
            current_label = base_string + str(i)
        else:
            current_label = string_list[i]

        ### Draw Circle: ###
        cv2.circle(image_frame, current_circle_point, current_circle_radius, (255, 255, 255), 3)
        image_frame = cv2.putText(image_frame, current_label, (
            current_circle_point[0] - current_circle_radius,
            current_circle_point[1] - current_circle_radius),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                  cv2.LINE_AA)

    return image_frame


def draw_circles_with_label_on_images(image_frames, circle_points, circle_radius_in_pixels=5, line_thickness=3, circles_label='Trj0'):
    #[image_frames] = [T,H,W] or [T,C,H,W]
    #circle_points = a 3D-array of size [N,3] with each row being (t_index, X_index, Y_index)  OR  a list of lists.
    #TODO: add the possibility of inserting a list of circle_radius_in_pixels for different sized circles
    for circle_index in np.arange(len(circle_points)):
        current_circle = circle_points[circle_index]
        t_index, x_index, y_index = current_circle
        current_image_frame = image_frames[int(t_index)]
        current_image_frame = draw_circle_with_label_on_image(current_image_frame,
                                                              (int(x_index), int(y_index)),
                                                              circle_label=circles_label,
                                                              circle_radius_in_pixels=circle_radius_in_pixels,
                                                              line_thickness=line_thickness)
        image_frames[int(t_index)] = current_image_frame
    return image_frames


def draw_circles_with_label_on_images_torch(image_frames, circle_points, circle_radius_in_pixels=5, line_thickness=3, circles_label='Trj0'):
    #[image_frames] = [T,H,W] or [T,C,H,W]
    #circle_points = a 3D-array of size [N,3] with each row being (t_index, X_index, Y_index)  OR  a list of lists.
    #TODO: add the possibility of inserting a list of circle_radius_in_pixels for different sized circles1
    for circle_index in np.arange(len(circle_points)):
        current_circle = circle_points[circle_index]
        t_index, x_index, y_index = current_circle
        current_image_frame = image_frames[int(t_index)]
        current_image_frame = draw_circle_with_label_on_image(current_image_frame,
                                                              (int(x_index), int(y_index)),
                                                              circle_label=circles_label,
                                                              circle_radius_in_pixels=circle_radius_in_pixels,
                                                              line_thickness=line_thickness)
        image_frames[int(t_index)] = current_image_frame
    return image_frames


def draw_trajectories_as_circle_on_images(image_frames, trajectories_list, circle_radius_in_pixels=5, line_thickness=3):
    # trajectories_list[0] = [N,3] with each row being (t_index, X_index, Y_index)  OR  a list of lists
    for trajectory_index in np.arange(len(trajectories_list)):
        current_circle_points = trajectories_list[trajectory_index]
        image_frames = draw_circles_with_label_on_images(image_frames,
                                                         current_circle_points,
                                                         circle_radius_in_pixels=circle_radius_in_pixels,
                                                         line_thickness=line_thickness,
                                                         circles_label='Trj'+str(trajectory_index))
    return image_frames

def draw_trajectories_as_circle_on_images_torch(image_frames, trajectories_list, circle_radius_in_pixels=5, line_thickness=3):
    # trajectories_list[0] = [N,3] with each row being (t_index, X_index, Y_index)  OR  a list of lists
    for trajectory_index in np.arange(len(trajectories_list)):
        current_circle_points = trajectories_list[trajectory_index]
        image_frames = draw_circles_with_label_on_images_torch(image_frames,
                                                         current_circle_points,
                                                         circle_radius_in_pixels=circle_radius_in_pixels,
                                                         line_thickness=line_thickness,
                                                         circles_label='Trj'+str(trajectory_index))
    return image_frames

def draw_circle_with_label_on_image(image_frame, circle_point, circle_label='', circle_radius_in_pixels=5, line_thickness=3):
    image_frame = cv2.circle(image_frame.copy(), circle_point, circle_radius_in_pixels, (255, 255, 255), line_thickness)
    image_frame = cv2.putText(image_frame, circle_label, (
        circle_point[0] - circle_radius_in_pixels,
        circle_point[1] - circle_radius_in_pixels),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                              cv2.LINE_AA)
    return image_frame

def draw_circle_with_label_on_image_torch(image_frame, circle_point, circle_label='', circle_radius_in_pixels=5, line_thickness=3):
    #TODO: add color
    ### Get Coordinates: ###
    X_center, Y_center = circle_point
    X_start = X_center - circle_radius_in_pixels
    X_stop = X_center + circle_radius_in_pixels
    Y_start = Y_center - circle_radius_in_pixels
    Y_stop = Y_center + circle_radius_in_pixels
    BB_tuple = (X_start, Y_start, X_stop, Y_stop)
    draw_bounding_boxes_with_labels_on_images_XYXY_torch(image_frame, BB_tuples_list_of_lists=[[BB_tuple]], color=torch.tensor([1]))
    return image_frame

def BB_convert_BB_notation_XYWH_to_XYXY(BB_tuple):
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]
    return (x0,y0,x0+W,y0+H)

def BB_convert_BB_notation_XYWH_to_XYXY(BB_tuple):
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    return (x0,y0,x1-x0,y1-y0)

def draw_bounding_box_with_label_on_image_XYWH(image_frame, BB_tuple=(0,0,100,100), BB_label='', line_thickness=3, color=(255,255,255), flag_draw_on_same_image=True):
    ### assumes BB_tuple = (x0,y0,W,H): ###

    ### Draw Bounding Box: ###
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    W = BB_tuple[2]
    H = BB_tuple[3]

    if flag_draw_on_same_image:
        image_frame_copy = image_frame
    else:
        image_frame_copy = copy.deepcopy(image_frame)

    cv2.rectangle(image_frame_copy, (x0,y0), (x0+W,y0+H), color, line_thickness)

    ### Draw text: ###
    image_frame_copy = cv2.putText(image_frame_copy, BB_label, (x0,y0),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                              cv2.LINE_AA)

    return image_frame_copy

def draw_bounding_boxes_with_labels_on_image_XYWH(image_frame, BB_tuples_list=[(0,0,100,100)], BB_labels_list=[''], line_thickness=3, color=(255,255,255), flag_draw_on_same_image=True):
    ### assumes BB_tuple = (x0,y0,W,H): ###
    if len(BB_labels_list) == 1:
        BB_labels_list = BB_labels_list * len(BB_tuples_list)
    for i in np.arange(len(BB_tuples_list)):
        BB_tuple = BB_tuples_list[i]
        BB_label = BB_labels_list[i]
        image_frame = draw_bounding_box_with_label_on_image_XYWH(image_frame, BB_tuple, BB_label, line_thickness=line_thickness, color=color, flag_draw_on_same_image=flag_draw_on_same_image)
    return image_frame

def draw_bounding_box_with_label_on_image_XYXY(image_frame, BB_tuple=(0,0,100,100), BB_label='', line_thickness=3, color=(255,255,255), flag_draw_on_same_image=True):
    ### Assumes BB_tuple = (x0,y0,x1,y1): ###

    ### Draw Bounding Box: ###
    x0 = BB_tuple[0]
    y0 = BB_tuple[1]
    x1 = BB_tuple[2]
    y1 = BB_tuple[3]
    W = x1 - x0
    H = y1 - y0

    if flag_draw_on_same_image:
        image_frame_copy = image_frame
    else:
        image_frame_copy = copy.deepcopy(image_frame)

    image_frame_copy = cv2.rectangle(image_frame_copy, (x0, y0), (x0 + W, y0 + H), color, line_thickness)

    ### Draw text: ###
    image_frame_copy = cv2.putText(image_frame_copy, BB_label, (x0, y0),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                              cv2.LINE_AA)

    return image_frame_copy

def draw_bounding_boxes_with_labels_on_image_XYXY(image_frame, BB_tuples_list=[(0,0,100,100)], BB_labels_list=[''], line_thickness=3, color=(255,255,255), flag_draw_on_same_image=True):
    ### assumes BB_tuple = (x0,y0,x1,y1): ###
    if len(BB_labels_list) == 1:
        BB_labels_list = BB_labels_list * len(BB_tuples_list)
    for i in np.arange(len(BB_tuples_list)):
        BB_tuple = BB_tuples_list[i]
        BB_label = BB_labels_list[i]
        image_frame = draw_bounding_box_with_label_on_image_XYXY(image_frame, BB_tuple, BB_label, line_thickness=line_thickness, color=color, flag_draw_on_same_image=flag_draw_on_same_image)
    return image_frame


def draw_bounding_boxes_with_labels_on_images_XYXY(image_frames, BB_tuples_list_of_lists=[[(0,0,100,100)]], BB_labels_list_of_lists=[['']], line_thickness=3, color=(255,255,255), flag_draw_on_same_image=True):
    ### assumes BB_tuple = (x0,y0,x1,y1): ###
    if len(BB_labels_list_of_lists) == 1:
        BB_labels_list_of_lists = BB_labels_list_of_lists * len(BB_tuples_list_of_lists)
    for i in np.arange(image_frames.shape[0]):
        current_image_frame = image_frames[i]
        current_BB_list = BB_tuples_list_of_lists[i]
        current_labels_list = BB_labels_list_of_lists[i]
        current_image_frame = draw_bounding_boxes_with_labels_on_image_XYXY(current_image_frame,
                                                                            current_BB_list,
                                                                            BB_labels_list=current_labels_list,
                                                                            line_thickness=line_thickness,
                                                                            color=color, flag_draw_on_same_image=flag_draw_on_same_image)
    return image_frames


def draw_bounding_boxes_with_labels_on_images_XYXY_torch(input_tensor, BB_tuples_list_of_lists=[[(0,0,100,100)]], color=torch.tensor([1])):
    ### assumes BB_tuple = (x0,y0,x1,y1): ###
    #(*). Note: draw_rectangle gets [BB_tensor] = [N_frames, B_bounding_boxes_per_frame, 4], with the last dim being XYXY
    BB_tensor = list_of_BB_lists_to_torch_tensor(BB_tuples_list_of_lists)
    input_tensor = kornia.utils.draw_rectangle(input_tensor, BB_tensor, color=torch.tensor([1]))
    return input_tensor

#TODO: add functionality to draw trajectories on torch tensors, to avoid memory and CPU usage!

def list_of_BB_lists_to_torch_tensor(list_of_BB, output_device='cuda'):
    ### Get Max Length Element: ###
    max_element_length = 0
    number_of_elements = len(list_of_BB)
    for i in np.arange(number_of_elements):
        current_BB_list = list_of_BB[i]
        max_element_length = max(max_element_length, len(current_BB_list))

    ### Initialize Tensor: ###
    BB_tensor = torch.zeros((number_of_elements, max_element_length, 4)).to(output_device)

    ### Loop Over and Fill Tensor: ###
    for i in np.arange(number_of_elements):
        current_BB_list = list_of_BB[i]
        for j in np.arange(len(current_BB_list)):
            BB_tensor[i, j, :] = torch.tensor(current_BB_list[j]).to(output_device)
            # BB_tensor[i, j, :] = current_BB_list[j]

    return BB_tensor


def draw_ellipse_with_label_on_image(image_frame, center_coordinates=(0,0), ellipse_angle=0, axes_lengths=(1,1), ellipse_label='', line_thickness=3, shape_color=(255,255,255), text_color=(0,0,255)):
    ### Initialize needed auxiliary variables: ###
    start_angle = 0
    stop_angle = 360

    ### Draw Ellipse: ###
    image_frame = cv2.ellipse(image_frame, center_coordinates, axes_lengths,
                        ellipse_angle, start_angle, stop_angle, shape_color, line_thickness)

    ### Put text: ###
    image_frame = cv2.putText(image_frame, ellipse_label, (
        center_coordinates[0] - axes_lengths[0],
        center_coordinates[1] - axes_lengths[1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1,
                              cv2.LINE_AA)
    return image_frame

#TODO: rewrite this by simply calling the above function
def draw_ellipses_with_labels_on_image(image_frame, center_coordinates_list=[(0,0)], ellipse_angles_list=[0], axes_lengths_list=[(1,1)], base_string='Trj', string_list=[], line_thickness=3):
    # circle_points = a list of tuples/list/array  OR  a 2D-array of size [N,2].   each circle_point is (X0,Y0) of the circle
    # TODO: add the possibility of inserting a list of circle_radius_in_pixels for different sized circles
    for i in np.arange(len(center_coordinates_list)):
        current_center_coordinates = center_coordinates_list[i]
        current_axes_lengths = axes_lengths_list[i]
        current_ellipse_angle = ellipse_angles_list[i]
        start_angle = 0
        stop_angle = 360
        if string_list == []:
            current_label = base_string + str(i)
        else:
            current_label = string_list[i]

        ### Draw Ellipse: ###
        image_frame = cv2.ellipse(image_frame, current_center_coordinates, current_axes_lengths,
                            current_ellipse_angle, start_angle, stop_angle, (255, 255, 255), line_thickness)

        ### Put text: ###
        image_frame = cv2.putText(image_frame, current_label, (
            current_center_coordinates[0] - current_axes_lengths[0],
            current_center_coordinates[1] - current_axes_lengths[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                  cv2.LINE_AA)

    return image_frame

def draw_text_on_image(image_frame, text_for_image='', org=(0,30), fontScale=0.7, color=(0,255,0), thickness=2):
    image_frame = cv2.putText(img=image_frame,
                                            text=text_for_image,
                                            org=org,
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=fontScale,
                                            color=color,
                                            thickness=thickness)
    return image_frame


#TODO: rewrite this function to wrap draw_circle_with_label_on_image for multiple frames
def draw_trajectories_on_images(input_frames, trajectory_tuple):
    #trajectory_tuple is an array of [N,3], where each tuple is (t,x,y)
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(input_frames,
                                                                    fps=50,
                                                                    tit='',
                                                                    Res_dir='',
                                                                    flag_save_movie=0,
                                                                    trajectory_tuple=trajectory_tuple)
    input_frames, frames_list = Plot_BoundingBox_On_Movie(input_frames,
                              fps=50,
                              tit='',
                              Res_dir='',
                              flag_save_movie=0,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

    return frames_list


def Plot_BoundingBoxes_On_Video(Mov, fps=1000, tit="tit", flag_transpose_movie=0, frame_skip_steps=1, resize_factor=1, flag_save_movie=0, Res_dir='Res_dir', histogram_limits_to_stretch =0.001, trajectory_tuple = []):
    # trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)

    ### Initialize Things: ###
    flag_found_trajectories = len(trajectory_tuple) == 3
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    number_of_frames_total,H,W = Mov.shape
    number_of_frames_to_view = number_of_frames_total // frame_skip_steps
    Mov = np.float32(Mov)
    wait1 = 1 / fps
    if flag_transpose_movie:
        iax0 = 1
        iax1 = 2
        Mov = np.transpose(Mov, (0,2,1))
        trajectory_smoothed_polynom_XY = (trajectory_tuple[2], trajectory_tuple[1])
    else:
        iax0 = 2
        iax1 = 1
        trajectory_smoothed_polynom_XY = (trajectory_tuple[1], trajectory_tuple[2])
    if flag_save_movie:
        # OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir,tit + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (resize_factor * Mov.shape[iax0], resize_factor * Mov.shape[iax1]), 0)
        OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir,tit + ".avi"), 0, fps, (resize_factor * Mov.shape[iax0], resize_factor * Mov.shape[iax1]), 0)

    ### Stretch Movie Histogram & Leave out the histogram_limits_to_stretch distribution ends on both sides: ###
    # histogram_limits_to_stretch_2 = 1 - histogram_limits_to_stretch
    # a, b = np.histogram(Mov[0], np.int(np.max(Mov[0])-np.min(Mov[0])))
    # a = np.cumsum(a)
    # a = a/np.max(a)
    # amn = b[np.where(a>histogram_limits_to_stretch)[0][0]]
    # amx = b[np.where(a>histogram_limits_to_stretch_2)[0][0]]
    # Mov[Mov>amx] = amx
    # Mov[Mov<amn] = amn
    # Mov = np.uint8(255*(Mov - amn)/(amx - amn))
    Mov = np.uint8(255*Mov.clip(0,1))

    ### Loop over movie frames and paint circle over discovered trajectories: ###
    BoundingBox_PerFrame_list = []
    for i in np.arange(number_of_trajectories):
        BoundingBox_PerFrame_list.append([])

    current_frame_index = 0
    for frame_index in np.arange(0,number_of_frames_total,frame_skip_steps):
        ### Get current movie frame: ###
        image_frame = Mov[current_frame_index, :, :]

        ### Resize Frame: ###
        new_size = tuple((np.array(image_frame.shape) * resize_factor).astype(int))
        image_frame = cv2.resize(image_frame, (new_size[1], new_size[0]), cv2.INTER_NEAREST)

        ### Loop Over Found Trajectories & Paint Circles On Them: ###
        for trajectory_index in range(number_of_trajectories):
            if (current_frame_index in t_vec[trajectory_index]):
                ### Find the index within current trajectory which includes current frame: ###
                current_frame_index_within_trajectory = np.where(current_frame_index == t_vec[trajectory_index])[0][0]

                ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
                target_spatial_coordinates = (np.int(trajectory_smoothed_polynom_XY[0][trajectory_index][current_frame_index_within_trajectory]*resize_factor),
                                              np.int(trajectory_smoothed_polynom_XY[1][trajectory_index][current_frame_index_within_trajectory]*resize_factor))

                ### Keep Track Of Bounding-Boxes (H_index, W_index): ###
                BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[1],target_spatial_coordinates[0]))

                ### Draw Circle On Screen: ###   #TODO: draw rectangle instead and register/output coordinates per frame cleanly for later analysis
                circle_radius_in_pixels = 10
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0,0,255), 1)
                image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (target_spatial_coordinates[0]-circle_radius_in_pixels, target_spatial_coordinates[1]-circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

        ### Actually Show Image With Circles On It: ###
        cv2.imshow(tit, image_frame)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write(image_frame)

        ### Skip Frame/s: ###
        current_frame_index += frame_skip_steps
        if ((cv2.waitKey(np.int(1000*wait1)) == ord('q')) | (current_frame_index>np.size(Mov,0)-1)):
            break

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()
    cv2.destroyWindow(tit)

    return BoundingBox_PerFrame_list


def Get_BoundingBox_List_For_Each_Frame(Mov, fps=1000, tit="tit", Res_dir='Res_dir', flag_save_movie=1, trajectory_tuple=[]):
    # trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)

    ### Initialize Things: ###
    # flag_found_trajectories = len(trajectory_tuple) == 3
    flag_found_trajectories = trajectory_tuple[0] != []
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    if len(Mov.shape) == 3:
        number_of_frames_total, H, W = Mov.shape
    elif len(Mov.shape) == 4:
        number_of_frames_total, H, W, C = Mov.shape
    number_of_frames_to_view = number_of_frames_total
    Mov = np.float32(Mov)
    wait1 = 1 / fps
    iax0 = 2
    iax1 = 1
    trajectory_smoothed_polynom_XY = (trajectory_tuple[1], trajectory_tuple[2])


    ### Stretch Movie Histogram & Leave out the histogram_limits_to_stretch distribution ends on both sides: ###
    Mov = np.uint8(255*Mov.clip(0,1))

    ### Loop over movie frames and paint circle over discovered trajectories: ###
    BoundingBox_PerFrame_list = []
    for i in np.arange(number_of_trajectories):
        BoundingBox_PerFrame_list.append([])

    ### Loop over movie frames and see if there are trajectories found in these frames: ###
    current_frame_index = 0
    for frame_index in np.arange(0,number_of_frames_total,1):
        ### Loop Over Found Trajectories & Paint Circles On Them: ###
        for trajectory_index in range(number_of_trajectories):
            if (current_frame_index in t_vec[trajectory_index]):
                ### Find the index within current trajectory which includes current frame: ###
                current_frame_index_within_trajectory = np.where(current_frame_index == t_vec[trajectory_index])[0][0]

                ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
                target_spatial_coordinates = (np.int(trajectory_smoothed_polynom_XY[0][trajectory_index][current_frame_index_within_trajectory]),
                                              np.int(trajectory_smoothed_polynom_XY[1][trajectory_index][current_frame_index_within_trajectory]))

                # ### Keep Track Of Bounding-Boxes (H_index, W_index): ###
                #TODO: make the entire script be (H,W) instead of (X,Y)
                # BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[1],target_spatial_coordinates[0]))
                ### Keep Track Of Bounding-Boxes (Xindex, Yindex): ###
                BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[0], target_spatial_coordinates[1]))
            else:
                ### If there isn't a drone trajectory found in current frame then simply put -1 instead: ###
                BoundingBox_PerFrame_list[trajectory_index].append(None)

        ### Skip Frame: ###
        current_frame_index += 1

    return BoundingBox_PerFrame_list


#TODO: better define this function - what do i input? tuple of [X,Y,H,W]? tuple of [X0,Y0, X1, Y1]? what exactly?!?!!?
def Plot_BoundingBox_On_Frame(image_frame, BoundingBox_list):
    ### Loop over trajectories and for each trajectory see if there exists a valid trajectory in current frame and if so then draw circle: ###
    number_of_trajectories = len(BoundingBox_list)
    for trajectory_index in range(number_of_trajectories):
        ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
        target_spatial_coordinates = BoundingBox_list[trajectory_index]

        if target_spatial_coordinates is not None:
            circle_radius_in_pixels = 10
            # TODO: stop mixing (X,Y) and (H,W). make the whole script be (H,W)!!!!
            # image_frame = cv2.circle(image_frame, (target_spatial_coordinates[1],target_spatial_coordinates[0]), circle_radius_in_pixels, (0, 0, 255), 1)
            if type(target_spatial_coordinates) != tuple:
                image_frame = cv2.circle(image_frame, tuple(target_spatial_coordinates), circle_radius_in_pixels, (0, 0, 255), 1)
            else:
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0, 0, 255), 1)
            image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                target_spatial_coordinates[0] - circle_radius_in_pixels,
                target_spatial_coordinates[1] - circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                      cv2.LINE_AA)

    return image_frame


def Plot_BoundingBox_On_Movie(Mov, fps=1000, tit="tit", Res_dir='Res_dir', flag_save_movie=1, trajectory_tuple=[], BoundingBox_PerFrame_list=[]):
    ### Initialize Things: ###
    flag_found_trajectories = trajectory_tuple[0] != []
    if flag_found_trajectories == False:
        return Mov, Mov

    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    if len(Mov.shape) == 3:
        number_of_frames_total, H, W = Mov.shape
    elif len(Mov.shape) == 4:
        number_of_frames_total, H, W, C = Mov.shape
    number_of_frames_to_view = number_of_frames_total
    iax0 = 2
    iax1 = 1
    if flag_save_movie:
        OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), cv2.VideoWriter_fourcc(*'MP42'), fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)

    ### Loop Over Video Frames & Paint Them On Screen: ###
    frames_list_with_BB = []
    for frame_index in np.arange(0, number_of_frames_total, 1):
        ### Get current movie frame: ###
        image_frame = np.copy(Mov[frame_index, :, :])

        ### Loop over trajectories and for each trajectory see if there exists a valid trajectory in current frame and if so then draw circle: ###
        for trajectory_index in range(number_of_trajectories):

            ### If There's Any Bounding Box Draw It: ###
            if BoundingBox_PerFrame_list != None:
                ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
                target_spatial_coordinates = BoundingBox_PerFrame_list[trajectory_index][frame_index]

                if target_spatial_coordinates is not None:
                    circle_radius_in_pixels = 10
                    #TODO: stop mixing (X,Y) and (H,W). make the whole script be (H,W)!!!!
                    # image_frame = cv2.circle(image_frame, (target_spatial_coordinates[1],target_spatial_coordinates[0]), circle_radius_in_pixels, (0, 0, 255), 1)
                    image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (255, 0, 0), 1)
                    image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                    target_spatial_coordinates[0] - circle_radius_in_pixels,
                    target_spatial_coordinates[1] - circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                              cv2.LINE_AA)

        ### Add frame to frames list: ###
        frames_list_with_BB.append(image_frame)
        ### Actually Show Image With Circles On It: ###
        # cv2.imshow(tit, image_frame)
        # plt.imshow(image_frame)
        # plt.pause(0.001)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write((image_frame*255).clip(0,255).astype(np.uint8))

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()

    return Mov, frames_list_with_BB


def Plot_BoundingBox_And_Polygon_On_Movie(Mov, fps=1000, tit="tit", Res_dir='Res_dir', flag_save_movie=1, trajectory_tuple=[],
                                          BoundingBox_PerFrame_list=[],
                                          polygon_points_list=[]):
    ### Initialize Things: ###
    flag_found_trajectories = len(trajectory_tuple) == 3
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    number_of_frames_total, H, W = Mov.shape
    number_of_frames_to_view = number_of_frames_total
    iax0 = 2
    iax1 = 1
    if flag_save_movie:
        # OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)
        OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), 0, fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)

    ### Loop Over Video Frames & Paint Them On Screen: ###
    for frame_index in np.arange(0, number_of_frames_total, 1):
        ### Get current movie frame: ###
        image_frame = Mov[frame_index, :, :]

        ### Loop over trajectories and for each trajectory see if there exists a valid trajectory in current frame and if so then draw circle: ###
        for trajectory_index in range(number_of_trajectories):
            ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
            target_spatial_coordinates = BoundingBox_PerFrame_list[trajectory_index][frame_index]

            if target_spatial_coordinates is not None:
                circle_radius_in_pixels = 10
                #TODO: stop mixing (X,Y) and (H,W). make the whole script be (H,W)!!!!
                # image_frame = cv2.circle(image_frame, (target_spatial_coordinates[1],target_spatial_coordinates[0]), circle_radius_in_pixels, (0, 0, 255), 1)
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0, 0, 255), 1)
                image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                                            target_spatial_coordinates[0] - circle_radius_in_pixels,
                                            target_spatial_coordinates[1] - circle_radius_in_pixels),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                          cv2.LINE_AA)

            ### Add Lines To Image: ###
            image_frame = draw_polygon_points_on_image(image_frame, polygon_points_list)

        ### Actually Show Image With Circles On It: ###
        # cv2.imshow(tit, image_frame)
        # plt.imshow(image_frame)
        # plt.pause(0.001)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write((image_frame*255).clip(0,255).astype(np.uint8))

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()

def Plot_Bounding_Box_Demonstration(Movie, prestring='', results_folder='', trajectory_tuple=[]):
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Movie,
                                                                    fps=50,
                                                                    tit=prestring + 'Movie_With_Drone_BB',
                                                                    Res_dir=results_folder,
                                                                    flag_save_movie=0,
                                                                    trajectory_tuple=trajectory_tuple)

    Movie_with_BB, frames_list = Plot_BoundingBox_On_Movie(Movie,
                              fps=50,
                              tit=prestring + "Movie_With_Drone_BB",
                              Res_dir=results_folder,
                              flag_save_movie=0,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

    return Movie_with_BB, torch.Tensor(frames_list)




def Save_Movies_With_Trajectory_BoundingBoxes(Movie, Movie_BGS, prestring='', results_folder='', trajectory_tuple=[]):
    # TODO: make sure it's only saved to disk and not on arrays themsleves!!!!
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Movie,
                                                                    fps=50,
                                                                    tit=prestring + 'Movie_With_Drone_BB',
                                                                    Res_dir=results_folder,
                                                                    flag_save_movie=0,
                                                                    trajectory_tuple=trajectory_tuple)
    Plot_BoundingBox_On_Movie(Movie,
                              fps=50,
                              tit=prestring + "Movie_With_Drone_BB",
                              Res_dir=results_folder,
                              flag_save_movie=1,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

    Plot_BoundingBox_On_Movie((Movie_BGS),  # notice this is the "normalized" BGS video
                              fps=50,
                              tit=prestring + "Movie_BGS_With_Drone_BB",
                              Res_dir=results_folder,
                              flag_save_movie=1,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)



def Plot_3D_PointCloud_With_Trajectories(res_points, NonLinePoints, xyz_line, num_of_trj_found, params, save_name,
                                         Res_dir, auto_close_flg=True):
    ### Get variables from params dict: ###
    SaveResFlg = params['SaveResFlg']
    roi = params['roi']
    T = params['FPS'] * params['SeqT']

    ### Stake All The Found Lines On Top Of Each Other: ###
    trajectory_TXY_stacked_numpy = np.zeros((0, 3))
    c = np.zeros((0, 1))
    for ii in range(num_of_trj_found):
        trajectory_TXY_stacked_numpy = np.vstack((trajectory_TXY_stacked_numpy, xyz_line[ii]))  #xyz_line are the points which belong to a trajectory!!!!
        c = np.vstack((c, ii * np.ones((xyz_line[ii].shape[0], 1))))
    c = c.reshape((len(c),))

    ### Create 3D Plot: ###
    fig = plt.figure(210)
    ax = plt.axes(projection='3d')

    ### Add all the outlier points found: ###
    if res_points.shape[1] > 0:
        ax.scatter(res_points[:, 0], res_points[:, 1], res_points[:, 2], c='k',
                   marker='*', label='Outlier', s=1)

    ### Add the "NonLinePoints" which were found by RANSAC but were disqualified after heuristics: ###
    ax.scatter(NonLinePoints[:, 0], NonLinePoints[:, 1], NonLinePoints[:, 2], c='k',
               marker='x', label='NonLinePoints', s=10)

    ### Add the proper trajectory: ###
    t_indices = trajectory_TXY_stacked_numpy[:, 0]
    x_indices = trajectory_TXY_stacked_numpy[:, 1]
    y_indices = trajectory_TXY_stacked_numpy[:, 2]
    scatter = ax.scatter(t_indices, x_indices, y_indices, c=c, marker='o', s=20)

    ### Add Legends: ###
    if trajectory_TXY_stacked_numpy.shape[0] != 0:
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Trj")
        ax.add_artist(legend1)

    # ### Maximize plot window etc': ###
    # ax.legend(loc='lower left')
    # ax.view_init(elev=20, azim=60)
    # plt.get_current_fig_manager().full_screen_toggle()
    # ax.set_xlim(0, T)
    # ax.set_zlim(0, roi[1])
    # ax.set_ylim(0, roi[0])
    # plt.pause(.000001)
    # plt.show()

    ### Save Figure If Wanted: ###
    if SaveResFlg:
        plt.savefig(os.path.join(Res_dir, save_name + ".png"))

    ### Close Plot: ###
    # if auto_close_flg:
    #     plt.close(fig)
    plt.pause(0.1)
    plt.close(fig)
    plt.close('all')
    plt.clf()
    plt.cla()


def Plot_3D_PointCloud_With_Trajectories_Demonstration(all_TXY_outliers,
                                                       NonLinePoints,
                                                       valid_TXY_outliers_trajectories_list,
                                                       num_of_trj_found,
                                                       params):
    ### Get variables from params dict: ###
    SaveResFlg = params['SaveResFlg']
    roi = params['roi']
    T = params['FPS'] * params['SeqT']

    ### Stake All The Found Lines On Top Of Each Other: ###
    trajectory_TXY_stacked_numpy = np.zeros((0, 3))
    c = np.zeros((0, 1))
    for ii in range(num_of_trj_found):
        trajectory_TXY_stacked_numpy = np.vstack((trajectory_TXY_stacked_numpy, valid_TXY_outliers_trajectories_list[ii]))  #xyz_line are the points which belong to a trajectory!!!!
        c = np.vstack((c, ii * np.ones((valid_TXY_outliers_trajectories_list[ii].shape[0], 1))))
    c = c.reshape((len(c),))

    ### Create 3D Plot: ###
    fig = plt.figure(210)
    ax = plt.axes(projection='3d')

    ### Add all the outlier points found: ###
    if all_TXY_outliers.shape[1] > 0:
        ax.scatter(all_TXY_outliers[:, 0], all_TXY_outliers[:, 1], all_TXY_outliers[:, 2], c='k',
                   marker='*', label='Outlier', s=1)

    ### Add the "NonLinePoints" which were found by RANSAC but were disqualified after heuristics: ###
    ax.scatter(NonLinePoints[:, 0], NonLinePoints[:, 1], NonLinePoints[:, 2], c='k',
               marker='x', label='NonLinePoints', s=10)

    ### Add the proper trajectory: ###
    t_indices = trajectory_TXY_stacked_numpy[:, 0]
    x_indices = trajectory_TXY_stacked_numpy[:, 1]
    y_indices = trajectory_TXY_stacked_numpy[:, 2]
    scatter = ax.scatter(t_indices, x_indices, y_indices, c=c, marker='o', s=20)

    ### Add Legends: ###
    if trajectory_TXY_stacked_numpy.shape[0] != 0:
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Trj")
        ax.add_artist(legend1)

    plt.xlabel('Time[sec]')
    plt.ylabel('X[pixels]')
    ax.set_zlabel('Y[pixels]')




    # ### Maximize plot window etc': ###
    # ax.legend(loc='lower left')
    # ax.view_init(elev=20, azim=60)
    # plt.get_current_fig_manager().full_screen_toggle()
    # ax.set_xlim(0, T)
    # ax.set_zlim(0, roi[1])
    # ax.set_ylim(0, roi[0])
    # plt.pause(.000001)
    # plt.show()


##############################################################################################################################################



    #########################################################################################################################################
####  Multiple Image Visualization Functions: ####
##################################################

class VizList(list):
    """Extended List class which can be binded to an matplotlib's pyplot axis
    and, when being appended a value, automatically update the figure.

    Originally designed to be used in a jupyter notebook with activated
    %matplotlib notebook mode.

    Example of usage:

    %matplotlib notebook
    from matplotlib import pyplot as plt
    f, (loss_axis, validation_axis) = plt.subplots(2, 1)
    loss_axis.set_title('Training loss')
    validation_axis.set_title('MIoU on validation dataset')
    plt.tight_layout()

    loss_list = VizList()
    validation_accuracy_res = VizList()
    train_accuracy_res = VizList()
    loss_axis.plot([], [])
    validation_axis.plot([], [], 'b',
                         [], [], 'r')
    loss_list.bind_to_axis(loss_axis)
    validation_accuracy_res.bind_to_axis(validation_axis, 0)
    train_accuracy_res.bind_to_axis(validation_axis, 1)

    Now everytime the list are updated, the figure are updated
    automatically:

    # Run multiple times
    loss_list.append(1)
    loss_list.append(2)


    Attributes
    ----------
    axis : pyplot axis object
        Axis object that is being binded with a list
    axis_index : int
        Index of the plot in the axis object to bind to

    """

    def __init__(self, *args):
        super(VizList, self).__init__(*args)
        self.object_count = 0
        self.object_count_history = []
        self.axis = None
        self.axis_index = None

    def append(self, object):
        self.object_count += 1
        self.object_count_history.append(self.object_count)
        super(VizList, self).append(object)
        self.update_axis()

    def bind_to_axis(self, axis, axis_index=0):
        self.axis = axis
        self.axis_index = axis_index

    def update_axis(self):
        self.axis.lines[self.axis_index].set_xdata(self.object_count_history)
        self.axis.lines[self.axis_index].set_ydata(self)
        self.axis.relim()
        self.axis.autoscale_view()
        self.axis.figure.canvas.draw()




def plot_multiple_images(X_images, filter_indices=None, flag_common_colorbar=1, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                    super_title=None, titles_string_list=None):
    #This assumes that the images within the image batch are plottable (i.e. with 1 channel as in grayscale or with 3 channels as in rgb)

    if type(X_images) == torch.Tensor:
        # transform to numpy:
        X_images = X_images.detach().numpy()
        if X_images.ndim == 4: #[N,C,H,W]:
            X_images = np.transpose(X_images,[0,2,3,1]) #--> [N,H,W,C]
        if X_images.ndim == 3: #[C,H,W] (only 1 image received)
            X_images = np.transpose(X_images,[1,2,0]) #-->[H,W,C]


    # Parameters:
    if filter_indices == None:
        filter_indices = arange(0, len(X_images))
    number_of_images_to_show = len(filter_indices)
    number_of_images_to_show_cols = int(ceil(sqrt(number_of_images_to_show)))
    number_of_images_to_show_rows = int(ceil(number_of_images_to_show / number_of_images_to_show_cols))

    if flag_common_colorbar == 0:
        # Simple SubPlots:
        fig = plt.figure()
        plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
        if super_title is not None:
            plt.suptitle(super_title)

        plot_counter = 0
        for filter_index in filter_indices:
            a = fig.add_subplot(number_of_images_to_show_rows, number_of_images_to_show_cols,
                                plot_counter + 1)
            current_image = X_images[filter_index]
            current_image = crop_center_percent(current_image,
                                                crop_percent)  # the "non-valid" convolution regions around the image overshadow the dynamic range in the image display so it can be a good idea to cut them out
            plt.imshow(current_image)
            if flag_colorbar == 1:
                plt.colorbar()
            if titles_string_list is not None:
                plt.title(titles_string_list[plot_counter])
            plot_counter += 1

    elif flag_common_colorbar == 1:
        # Advanced SubPlots:
        fig, axes_array = plt.subplots(number_of_images_to_show_rows, number_of_images_to_show_cols)
        axes_array = np.atleast_2d(axes_array)
        fig.suptitle(super_title)
        print(shape(axes_array))
        plot_counter = 0
        image_plots_list = []
        cmap = "cool"
        for i in range(number_of_images_to_show_rows):
            for j in range(number_of_images_to_show_cols):
                if plot_counter >= number_of_images_to_show:
                    break
                # Generate data with a range that varies from one plot to the next.
                filter_index = filter_indices[plot_counter]
                current_image = X_images[filter_index]
                current_image = crop_center_percent(current_image, crop_percent)
                image_plots_list.append(axes_array[i, j].imshow(current_image, cmap=cmap))
                axes_array[i, j].label_outer()
                # if titles_string_list is not None:
                #     image_plots_list[plot_counter].title(titles_string_list[plot_counter])
                plot_counter += 1

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in image_plots_list)
        vmax = max(image.get_array().max() for image in image_plots_list)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for image in image_plots_list:
            image.set_norm(norm)

        fig.colorbar(image_plots_list[0], ax=axes_array, orientation='horizontal', fraction=.1)





def plot_multiple_batches_of_images(X_images, filter_indices=None, flag_common_colorbar=1, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                                    super_title=None, titles_string_list=None):
    #Assumes X_images is in the form of: [N,H,W] or [N,H,W,C]
    for i in arange(len(X_images)):
        plot_multiple_images(X_images[i], filter_indices, flag_common_colorbar, crop_percent, delta, flag_colorbar, super_title, titles_string_list)



def plot_multiple_image_channels(X_images, filter_indices=None, flag_common_colorbar=1, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                                    super_title=None, titles_string_list=None):
    #Assumes X_images is in the form of: [N,H,W] or [N,H,W,C]
    if X_images.ndim() == 4:
        number_of_images = X_images.shape[0]
    else:
        number_of_images = 1

    #Plot each image's channels in different figures:
    for i in arange(number_of_images):
        plot_multiple_images(X_images[i], filter_indices, flag_common_colorbar, crop_percent, delta, flag_colorbar, super_title, titles_string_list)


##############
def get_image_differences(imageA, imageB, flag_plot=1, crop_percent=0.8, flag_normalize_by_number_of_channels=1):
    #Remember, input image ranges can be [0,1] and [0,255] depending on whether float or uint8
    grayA = convert_to_grayscale(imageA,flag_scale_to_normal_range=0)
    grayB = convert_to_grayscale(imageB,flag_scale_to_normal_range=0)
    if flag_normalize_by_number_of_channels == 1:
        grayA = grayA / imageA.shape[2]
        grayB = grayB / imageB.shape[2]

    imageA = np.atleast_3d(imageA)
    flag_input_image_plotable = (imageA.shape[2] == 3 or imageA.shape[2] == 1)
    if flag_input_image_plotable:  # BW and RGB style images can be visualized... otherwise put the rectangle on the synthetic grayscale summed image
        imageA_with_rectangle = imageA.copy()  # Should i use .copy() or not??
        imageB_with_rectangle = imageB.copy()
        raw_diff = imageA - imageB
    else:
        imageA_with_rectangle = grayA.copy()
        imageB_with_rectangle = grayB.copy()
        raw_diff = grayA - grayB

    #Use compare_ssim and get overall score and pixel-wise diff which is NOT the RAW diff:
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff_scaled = (diff * 255).astype("uint8")
    # print(diff.shape)
    thresh = cv2.threshold(diff_scaled, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[1]

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA_with_rectangle, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB_with_rectangle, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if flag_plot == 1:
        if flag_input_image_plotable==False:
            plot_multiple_images([raw_diff, grayA, grayB, imageA_with_rectangle, imageB_with_rectangle, diff, thresh], flag_common_colorbar=0,
                            crop_percent=crop_percent, titles_string_list=['raw_diff', 'imageA_gray', 'imageB_gray', 'imageA_gray_rect', 'imageB_gray_rect', 'SSIM_map', 'threshold'])
        if flag_input_image_plotable==True:
            plot_multiple_images([raw_diff, imageA, imageB, imageA_with_rectangle, imageB_with_rectangle, diff, thresh], flag_common_colorbar=0,
                            crop_percent=crop_percent, titles_string_list=['raw_diff', 'imageA', 'imageB', 'imageA_rect', 'imageB_rect', 'SSIM_map', 'threshold'])

    return raw_diff, diff, thresh, imageA_with_rectangle, imageB_with_rectangle
################



def spot_check_images_from_batch_matrix(X_images, number_of_images_to_show, flag_rgb_or_gray=1, flag_colorbar=0, super_title=None, titles_string_list=None):
    number_of_images_to_show_each_axis = int(ceil(sqrt(number_of_images_to_show)))
    image_indices_to_show = np.random.choice(arange(0, shape(X_images)[0]), number_of_images_to_show, replace=False)

    fig = plt.figure()
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    if super_title is not None:
        plt.suptitle(super_title)
    for x_counter in arange(0, number_of_images_to_show_each_axis):
        for y_counter in arange(0, number_of_images_to_show_each_axis):
            current_index = x_counter * number_of_images_to_show_each_axis + y_counter
            if current_index >= number_of_images_to_show:
                return
            a = fig.add_subplot(number_of_images_to_show_each_axis, number_of_images_to_show_each_axis,
                                current_index + 1)
            if flag_rgb_or_gray == 1:
                plt.imshow(X_images[current_index,:])
            else:
                plt.imshow(skimage.color.rgb2gray(X_images[current_index, :]))
            if flag_colorbar == 1:
                plt.colorbar()
            if titles_string_list is not None:
                plt.title(titles_string_list[current_index])
#########################################################################################################################################





#########################################################################################################################################
####  Multiple Graphs Visualization Functions: ####
##################################################
from RapidBase.Utils.MISCELENEOUS import get_n_colors
def plot_multiple(y_plots_list, x_vec=None, number_of_samples_to_show=None, legend_labels=None, super_title=None, x_label=None, y_label=None):
    fig, axes_array = plt.subplots(1, 1)
    delta = 0.02
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    plt.figlegend(legend_labels, loc='lower center', ncol=5, labelspacing=0.)
    # current_axes = axes_array[0, 0]
    number_of_plots = len(y_plots_list)
    colors = get_n_colors(number_of_plots)

    ### create x_vec if needed: ###
    if x_vec is None:
        x_vec = np.arange(len(y_plots_list[0]))

    ### number of samples to show: ###
    if number_of_samples_to_show is None:
        number_of_samples_to_show = len(y_plots_list[0])

    ### Add super title: ###
    if super_title is not None:
        plt.suptitle(super_title)

    ### Plot all sub plots: ###
    for plot_counter in np.arange(0, number_of_plots):
        plt.plot(x_vec, y_plots_list[plot_counter][0:number_of_samples_to_show], c=colors[plot_counter])

    ### Add Legends: ###
    plt.legend(loc=2, bbox_to_anchor=[0, 1], labels=legend_labels, ncol=2, shadow=True, fancybox=True)

    ### Y limits So Legend Won't Block Graphs: ###
    y_min_value = np.inf
    y_max_value = -np.inf
    for plot_counter in np.arange(0, number_of_plots):
        y_min_value = min(y_min_value, y_plots_list[plot_counter][0:number_of_samples_to_show].min())
        y_max_value = max(y_max_value, y_plots_list[plot_counter][0:number_of_samples_to_show].max())
    plt.ylim([y_min_value, y_max_value * 1.3])

    ### X label and Y label: ###
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)




def plot_subplots(y_plots, flag_plot_or_histogram=1, number_of_samples_to_show=None, legend_labels=None,
                  super_title=None, x_label=None,y_label=None,title_strings=None):
    # y_plots[0:number_of_samples_to_show, current_subplot_index, plot_counter]

    if length(np.shape(y_plots)) == 2:
        y_plots = np.expand_dims(y_plots, -1)
    number_of_plots_each_figure_axes = np.shape(y_plots)[2]
    number_of_images_to_show = np.shape(y_plots)[1]
    number_of_images_to_show_x = int(np.ceil(np.sqrt(number_of_images_to_show)))
    number_of_images_to_show_y = int(np.ceil(number_of_images_to_show / number_of_images_to_show_x))
    if number_of_samples_to_show == None:
        number_of_samples_to_show = np.shape(y_plots)[0]

    fig, axes_array = plt.subplots(number_of_images_to_show_x, number_of_images_to_show_y)
    delta = 0.02
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    plt.figlegend(legend_labels, loc='lower center', ncol=5, labelspacing=0.)

    if super_title is not None:
        plt.suptitle(super_title)
    for x_counter in np.arange(0, number_of_images_to_show_x):
        for y_counter in np.arange(0, number_of_images_to_show_y):
            current_subplot_index = x_counter * number_of_images_to_show_y + y_counter
            if current_subplot_index >= number_of_images_to_show:
                axes_array[x_counter, y_counter].set_visible(False)
            else:
                current_axes = axes_array[x_counter, y_counter]

                #Plot simple plot or histogram:
                if flag_plot_or_histogram == 1:
                    for plot_counter in np.arange(0, number_of_plots_each_figure_axes):
                        current_axes.plot(y_plots[0:number_of_samples_to_show, current_subplot_index, plot_counter])
                elif flag_plot_or_histogram == 2:
                    current_axes.hist(y_plots[0:number_of_samples_to_show, current_subplot_index, :], alpha=0.5)


                if current_subplot_index == 0:
                    current_axes.legend(loc=2, bbox_to_anchor=[0, 1], labels=legend_labels, ncol=2, shadow=True, fancybox=True)
                if x_label is not None:
                    current_axes.set_xlabel(x_label)
                if y_label is not None:
                    current_axes.set_ylabel(y_label)
                if title_strings is not None:
                    current_axes.set_title(title_strings[current_subplot_index])
    return 1
############################################################################################################################################################






############################################################################################################################################################
##### Feature Distribution Visualization Functions: ######
def plot_bar_graph(x_positions=None, y_heights=None, x_labels=None, y_label=None, title_str=None):
    number_of_bars = length(x_positions)
    color_array = get_spaced_colors(number_of_bars)
    fig, ax = plt.subplots()
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    plt.show(block=False)
    bar_array = bar(x_positions, y_heights, width=0.3, align='center')
    for counter in arange(0, length(bar_array)):
        bar_array[counter].set_facecolor(color_array[counter])
    ax.set_xticks(np.arange(0, number_of_bars))

    if 'x_labels' in locals().keys():
        ax.set_xticklabels(x_labels)
    if 'y_label' in locals().keys():
        ax.set_ylabel(y_label)
    if 'title_str' in locals().keys():
        ax.set_title(title_str)


def plot_pair_plot(df,super_title):
    pair_plot = sns.pairplot(df)
    pair_plot_fig = pair_plot.fig
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    if 'super_title' in locals().keys():
        pair_plot_fig.suptitle(super_title, fontsize=14)



def plot_box_plot(df,x_labels,y_label,title_str,**kwargs):
    fig, ax = plt.subplots()
    plt.show(block=False)
    plt.boxplot(np.asarray(df),**kwargs)
    if 'x_labels' in locals().keys():
        ax.set_xticklabels(x_labels)
    if 'y_label' in locals().keys():
        ax.set_ylabel(y_label)
    if 'title_str' in locals().keys():
        ax.set_title(title_str)


# ###### Example Of Use: #######
# different_classes_vec = np.unique(y_class_labels_train).astype(int)
# number_of_classes = length(different_classes_vec)
# [unique_class_values_sorted_vec, original_array_with_classes_instead_of_values, class_occurence_vec] = \
#                       np.unique(y_class_labels_train,return_index=False, return_inverse=True, return_counts=True, axis=None)
# plot_bar_graph(unique_class_values_sorted_vec,class_occurence_vec,classification_output_name_list,'Class Occurence','Class Imbalance Bar Graph')
#
# plot_pair_plot(df_y_geometric_labels_train_only_ellipse,'Pair Plots of Geometric Labels (Ellipse Only)')
# plot_box_plot(df_y_geometric_labels_train_only_ellipse,y_total_labels_line_format_train_split[1:],'Distribution','Geometric Labels Box Plots, Train Data - Before PreProcessing',
#                   notch=True, sym=None,patch_artist=True, meanline=True, showmeans=True, showcaps=True, showbox=True, manage_xticks=True)
############################################################################################################################################################



############################################################################################################################################################
##### Classification Model Evaluation Functions: ######
def plot_multi_class_ROC(y_class_labels_test, y_prediction_labels):
    #Take Care of Input:
    if np.asarray(shape(y_class_labels_test)).prod() == length(y_class_labels_test):
        #If 1D verctor this means 2 classes -> turn to categorical
        y_class_labels_test = keras.utils.to_categorical(y_class_labels_test,2)
        y_prediction_labels = keras.utils.to_categorical(y_prediction_labels,2)

    # Compute ROC and AUC for each class:
    false_positive_rates = dict()
    true_positive_rates = dict()
    roc_auc = dict()
    for i in arange(0, number_of_classes):
        false_positive_rates[i], true_positive_rates[i], _ = roc_curve(y_class_labels_test[:, i],
                                                                       y_prediction_labels[:, i])
        roc_auc[i] = auc(false_positive_rates[i], true_positive_rates[i])
    # Compute micro-average ROC curve & AUC:
    false_positive_rates['micro'], true_positive_rates['micro'], _ = roc_curve(y_class_labels_test.ravel(),
                                                                               y_prediction_labels.ravel())
    roc_auc['micro'] = auc(false_positive_rates['micro'], true_positive_rates['micro'])
    # Compute macro-average ROC curve & AUC:
    #   (*). Aggregate all false positive rates:
    all_false_positive_rates = np.unique(
        np.concatenate([false_positive_rates[i] for i in arange(0, number_of_classes)]))
    #   (*). Interpolate all ROC curves at these points:
    mean_true_positive_rates = np.zeros_like(all_false_positive_rates)
    for i in arange(0, number_of_classes):
        mean_true_positive_rates += interp(all_false_positive_rates, false_positive_rates[i], true_positive_rates[i])
    #   (*). Average it and compute AUC:
    mean_true_positive_rates /= number_of_classes
    #   (*). Assign macro {key,value} to false_positive_rates dictionary:
    false_positive_rates['macro'] = all_false_positive_rates
    true_positive_rates['macro'] = mean_true_positive_rates
    roc_auc['macro'] = auc(false_positive_rates['macro'], true_positive_rates['macro'])
    # Plot all ROC curves
    lw = 2
    plt.figure(1)
    plt.plot(false_positive_rates["micro"], true_positive_rates["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(false_positive_rates["macro"], true_positive_rates["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(number_of_classes), colors):
        plt.plot(false_positive_rates[i], true_positive_rates[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC to multi-class')
    plt.legend(loc="lower right")
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f} misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2: (len(lines) - 3)]:
        # print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')


# #########      Example Of Use:     ############
# #Get Prediction Results
# tic_toc.tic()
# K.set_learning_phase(0)
# y_prediction_class_labels = final_classification_model.predict(X_test)
# K.set_learning_phase(1)
# y_prediction_class_labels_uncategorical = argmax(y_prediction_class_labels,axis=1)
# y_class_labels_test_uncategorical = argmax(y_class_labels_test,axis=1)
# tic_toc.toc(True)
# #Visualize Classification Results:
# #(1). Confusion Matrix:
# plt.figure()
# confusion_matrix_mat = confusion_matrix(y_class_labels_test_uncategorical,y_prediction_class_labels_uncategorical)
# plot_confusion_matrix(confusion_matrix_mat,classification_output_name_list,title='Confusion matrix',cmap=None,normalize=True)
# #(2). Classification Report:
# print(classification_report(y_class_labels_test_uncategorical,y_prediction_class_labels_uncategorical,target_names=['Line','Ellipse']))
# plt.figure()
# plot_classification_report(classification_report(y_class_labels_test_uncategorical,y_prediction_class_labels_uncategorical,target_names=['Line','Ellipse']))
# #(3). ROC:
# plot_multi_class_ROC(y_class_labels_test,y_prediction_class_labels)
# #(4). Spot Check Images Misclassified:
# for class_counter in arange(0,number_of_classes):
#     #Get Indices of instances Which are of the current class but were MISCLASSIFIED as another class:
#     indices_current_class_misclassified = indices_from_logical(
#                                                 (y_prediction_class_labels_uncategorical!=class_counter) &
#                                                 (y_class_labels_test_uncategorical==class_counter))
#     #Show Images Misclassified:
#     number_of_images_to_show = 10
#     number_of_images_to_show = np.minimum(number_of_images_to_show,length(indices_current_class_misclassified))
#     if number_of_images_to_show > 0:
#         image_indices_to_show = np.random.choice(indices_current_class_misclassified, number_of_images_to_show, replace=False)
#         current_super_title = 'Images of Class ' + classification_output_name_list[class_counter] + ' Misclassified as another class'
#         spot_check_images_from_batch_matrix((X_test[image_indices_to_show,:]*255+155).astype(int16),number_of_images_to_show,1,0,current_super_title)
#########################################################################################################################################










### Colors: ###
def get_color_formula(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(2*pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(2*pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(2*pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(2*pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(4*pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet

def gray2color(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]
    formula_triplet = get_color_formula(formula_id_triplet)
    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clamp(0,1)
    G = G.clamp(0,1)
    B = B.clamp(0,1)
    color_array = torch.cat([R,G,B], dim=1)
    return color_array

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,1,100,100)).abs().clamp(0,1)
# color_tensor = gray2color(input_tensor,0)
# figure(1)
# imshow_torch(input_tensor[0].repeat((3,1,1)),0)
# figure(2)
# imshow_torch(color_tensor[0],0)



















### Colors: ###
def get_color_formula_torch(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(2*pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(2*pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(2*pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(2*pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(4*pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet




def get_color_formula_numpy(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.cos(pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.cos(pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.sin(2*pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: numpy.cos(2*pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.sin(2*pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.cos(2*pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(numpy.sin(4*pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet


def gray2color_numpy(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]

    formula_triplet = get_color_formula_numpy(formula_id_triplet)

    input_min = input_array.min()
    input_max = input_array.max()
    input_array = to_range(input_array,0,1)
    # input_array = input_array/256

    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clip(0,1)
    G = G.clip(0,1)
    B = B.clip(0,1)
    if len(R.shape)==4:
        color_array = numpy.concatenate([R,G,B], 3)
    else:
        color_array = numpy.concatenate([R,G,B], 2)

    # input_array = input_array*256
    color_array = to_range(color_array, input_min, input_max)




    return color_array



def gray2color_torch(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]
    formula_triplet = get_color_formula_torch(formula_id_triplet)
    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clamp(0,1)
    G = G.clamp(0,1)
    B = B.clamp(0,1)

    if len(R.shape)==4:
        color_array = torch.cat([R,G,B], dim=1)
    else:
        color_array = torch.cat([R,G,B], dim=0)

    return color_array



def to_range(input_array, low, high):
    new_range_delta = high-low
    old_range_delta = input_array.max() - input_array.min()
    new_min = low
    old_min = input_array.min()
    input_array = ((input_array-old_min)*new_range_delta/old_range_delta) + new_min
    return input_array


