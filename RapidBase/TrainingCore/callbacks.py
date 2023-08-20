import os
import torch
import cv2
import numpy as np
from RapidBase.Utils.IO.Path_and_Reading_utils import save_image_torch, save_image_numpy
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch
import scipy.io
from RapidBase.Utils.ML_and_Metrics.Metrics import get_metrics_video_lists
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_to_numpy, crop_torch_batch, crop_numpy_batch,\
    torch_get_4D, torch_get_3D, torch_get_2D, torch_get_5D, numpy_array_to_video_ready, BW2RGB
# from RapidBase.import_all import *

### Some Callbacks For Your Convenience: ###
# InferenceCallback_ValidationDataSet(GeneralCallback): # Save original, noisy and clean. mainly used for validation or when i want to get statistics over a dataset in a single folder
# InferenceCallback_Movie(GeneralCallback): # this is relevant, for instance, when creating a cleaned version of a movie and i want to create a folder with only those frames
# InferenceCallback_Movie_ExtraDataset(GeneralCallback): # the same as InferenceCallback_Movie but relevant for when there is a clean_movie_test_dataset instead of using only one dataset object and adding noise
# InferenceCallback_Denoise # when there's a simple folder from which i do the debug/validation
# InferenceCallback_Denoise_OnlyStats # the same as InferenceCallback_Denoise but doesn't save images, only stats
# InferenceCallback_Denoise_ExternalOriginal # the same as InferenceCallback_Denoise but with a clean_folder_test_dataset instead of using only one dataset object and adding noise
# InferenceCallback_Denoise_WithRunningMean # the same as InferenceCallback_Denoise but there also uses "running_mean" from dataset object or wherever to compare to simple running mean
# InferenceCallback_Denoise_Recursive # saves all images from a recursive network in the recursive process (if i set, for instance, 25 images, it will record all of those + statistics)
# InferenceCallback_Denoise_Recursive_ClassicIIR # again, a callback for recursion but more suitable for ClassicIIR outputs
# InferenceCallback_Denoise_WithActualMean # self explanatory


class GeneralCallback:
    def __init__(self, output_path, Train_dict=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            # ans = input(f"{output_path} doesn't exists, should I create [Y/N]: ")
            # if ans == 'Y':
            #     os.makedirs(output_path)
            # else:
            #     raise OSError(f"Path {output_path} doesn't exist")
        self.path = output_path
        pass

    def run(self, index, inputs_dict, outputs):
        raise NotImplementedError("Subclass must implement this method")



class DumpCallback(GeneralCallback):
    def __init__(self, output_path):
        super().__init__(output_path)
        pass

    def run(self, index, inputs_dict, outputs):
        self.dump(index, inputs_dict, outputs)
        pass

    def dump(self, index, inputs_dict, outputs):
        torch.save({
            'inputs_dict': inputs_dict,
            'outputs': outputs['magic_output'].cpu().detach().numpy(),
            'model_state': outputs['model_state'],
        }, os.path.join(self.path, 'model_dump' + string_rjust(index, 6) + '.pth.tar'))

    def l1(self, a, b):
        return np.sum(np.abs(a - b))

    def compare(self, dump1_path, dump2_path):
        dump1 = torch.load(dump1_path)
        dump2 = torch.load(dump2_path)

        diff_outputs = self.l1(dump1['outputs'], dump2['outputs'])
        if diff_outputs != 0:
            raise ValueError(f"There is a diff of {diff_outputs} in the outputs")

        input1 = dump1['inputs_dict']
        input2 = dump2['inputs_dict']

        for key in input1.keys():
            a = np.int32(input1[key].cpu().numpy())
            b = np.int32(input2[key].cpu().numpy())
            diff = self.l1(a, b)
            if diff != 0:
                raise ValueError(f"Found inputs_dict diff of {diff} for key {key}")

        model1 = dump1['model_state']
        model2 = dump2['model_state']

        for key in model1.keys():
            a = np.int32(model1[key].cpu().numpy())
            b = np.int32(model2[key].cpu().numpy())
            diff = self.l1(a, b)
            if diff != 0:
                raise ValueError(f"Found model diff of {diff} for key {key}")
        print("Compare completed, everything is OK")
        pass


### Normal Training: ###
class InferenceCallback(GeneralCallback):
    # Save original, noisy and clean. mainly used for validation or when i want to get statistics over a dataset in a single folder
    def __init__(self, output_path, Train_dict=None):
        super().__init__(output_path, Train_dict)
        self.is_training = Train_dict.is_training
        pass

    def run(self, index, inputs_dict, outputs_dict):

        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)

    def save_outputs(self, outputs, index):
        # Disparity values are scaled by 32
        path = self.path
        clean_estimate_image = np.uint16(outputs['clean_frame_estimate'][0].cpu().detach().numpy().transpose([1, 2, 0]))

        save_image_torch(path, string_rjust(index, 6) + '_clean_frame_estimate.png',
                         torch_tensor=outputs['clean_frame_estimate'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        # scipy.io.savemat(path + '\Disparity_DS5.mat', mdict={
        #     'Disparity_DS5': inputs_dict['left_disparity_input'][0].cpu().numpy().transpose(1, 2, 0)})

    def save_inputs(self, inputs_dict, index):
        # Disparity values are scaled by 32
        path = self.path
        center_frame_original = np.uint16(inputs_dict['center_frame_original'][0].cpu().detach().numpy().transpose([1, 2, 0]))
        center_frame_noisy = np.uint16(inputs_dict['center_frame_noisy'][0].cpu().detach().numpy().transpose([1, 2, 0]))

        save_image_torch(path, string_rjust(index, 6) + '_original_image.png',
                         torch_tensor=inputs_dict['center_frame_original'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False,
                         flag_scale_by_255=True,
                         flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
        save_image_torch(path, string_rjust(index, 6) + '_noisy_image.png',
                         torch_tensor=inputs_dict['center_frame_noisy'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False,
                         flag_scale_by_255=True,
                         flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)


    def rgb2bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


### Denoising Callback Base: ###
#TODO: use the above callback "InferenceCallback" as the basic/father callback with the all proper functions.
#TOOD: i shold be able to unify all the callbacks into two callbacks i think, recursive and non-recursive
class InferenceCallback_Denoising_Base(GeneralCallback):
    def __init__(self, output_path, Train_dict=None):
        super().__init__(output_path, Train_dict)
        self.is_training = Train_dict.is_training

        ### Initialize dictionaries which keep track of the results: ###
        self.original_noisy_average_metrics_dict = AverageMeter_Dict()
        self.original_cleaned_average_metrics_dict = AverageMeter_Dict()
        self.original_running_mean_average_metrics_dict = AverageMeter_Dict()
        self.original_noisy_history_metrics_dict = KeepValuesHistory_Dict()
        self.original_cleaned_history_metrics_dict = KeepValuesHistory_Dict()
        self.original_running_mean_history_metrics_dict = KeepValuesHistory_Dict()
        self.avg_psnr_noisy_list = []
        self.avg_ssim_noisy_list = []
        self.avg_psnr_clean_list = []
        self.avg_ssim_clean_list = []
        self.steps = []
        ### Paths: ###
        self.inputs_original_path = self.path
        self.inputs_noisy_path = self.path
        self.inputs_temporal_average_path = self.path
        self.outputs_clean_path = self.path
        self.spatial_blur_path = self.path
        self.outputs_everything_concat_path = self.path
        self.statistics_path = self.path
        self.movie_path = self.path

        self.Train_dict = Train_dict
        self.Video_Writer_Original_Noisy = None
        pass


    def set_internal_variables_T(self, index, inputs_dict, outputs_dict, Train_dict=None):
        self.normalization_stretch_factor = 1 / (inputs_dict['center_frame_noisy'].cpu().max() + 1e-3) * 0.9
        self.normalization_stretch_factor = torch.Tensor(self.normalization_stretch_factor.cpu().numpy())
        self.normalization_stretch_factor_numpy = self.normalization_stretch_factor.cpu().numpy()

        ### Original and Noisy Frames: ###
        self.output_frames_noisy = torch_get_5D(inputs_dict['output_frames_noisy_HR'][0])
        self.output_frames_original = torch_get_5D(inputs_dict['output_frames_original'][0])
        self.center_frame_original = torch_get_5D(inputs_dict['center_frame_original'][0])
        self.center_frame_noisy = torch_get_5D(inputs_dict['center_frame_noisy_HR'][0])
        self.center_frame_actual_mean = torch_get_5D(inputs_dict['center_frame_actual_mean'][0])
        self.clean_frame_estimate = torch_get_5D(outputs_dict['center_clean_frame_estimate_for_callback'][0])

        ### Normalize Direct Inputs/Outputs: ###
        self.model_output = torch_get_5D(outputs_dict['model_output_for_callback'][0])

        ### Running Average Stuff: ###
        # self.center_frame_actual_mean_torch_stretched = (self.center_frame_actual_mean.cpu() * self.normalization_stretch_factor)
        self.center_frame_actual_mean_torch_stretched = (self.output_frames_noisy.cpu() * self.normalization_stretch_factor).mean(1,True)
        if 'current_moving_average' in inputs_dict.keys():
            self.current_moving_average = torch_get_5D(inputs_dict['current_moving_average'][0])
            self.center_frame_noisy_running_mean_numpy = torch_to_numpy(self.current_moving_average)
            self.current_moving_average_torch_stretched = (self.normalization_stretch_factor * inputs_dict['current_moving_average']) #TODO: make sure this is correct

        ### Clean Frame Estimate: ###
        self.clean_frame_estimate_numpy = (torch_to_numpy(self.clean_frame_estimate))
        self.clean_frame_estimate_numpy_stretched = (self.normalization_stretch_factor_numpy * self.clean_frame_estimate_numpy)
        self.clean_frame_estimate_torch_stretched = (self.normalization_stretch_factor * self.clean_frame_estimate)

        ### Center Original: ###
        if 'current_gt_clean_frame' in inputs_dict.keys():
            #TODO: seperate the recursive from the non-recursive in all scripts to avoid condition here
            #(1). Recursive Network:
            self.center_frame_original = torch_get_5D(inputs_dict['current_gt_clean_frame'][0])
        else:
            #(2). Non-Recursive Network:
            self.center_frame_original = torch_get_5D(inputs_dict['center_frame_original'][0])
        self.center_frame_original_numpy = torch_to_numpy(self.center_frame_original)
        self.center_frame_original_numpy_stretched = (self.center_frame_original_numpy * self.normalization_stretch_factor_numpy)
        self.center_frame_original_torch_stretched = (self.center_frame_original * self.normalization_stretch_factor)

        ### Center Noisy: ###
        self.center_frame_noisy_numpy = torch_to_numpy(self.center_frame_noisy)
        self.center_frame_noisy_numpy_stretched = self.normalization_stretch_factor_numpy * self.center_frame_noisy_numpy
        self.center_frame_noisy_torch_stretched = (self.center_frame_noisy * self.normalization_stretch_factor)
        
        ### Recursive Stuff: ###
        if 'current_noisy_frame' in inputs_dict.keys():
            self.current_noisy_frame_torch_stretched = torch_get_5D(self.normalization_stretch_factor * inputs_dict['current_noisy_frame'][0])
        if 'current_gt_clean_frame' in inputs_dict.keys():  #TODO: i think i can delete this already
            self.current_gt_clean_frame_torch_stretched = torch_get_5D(self.normalization_stretch_factor * inputs_dict['current_gt_clean_frame'][0])
                

    def set_internal_variables(self, index, inputs_dict, outputs_dict, Train_dict):
        self.set_internal_variables_T(index, inputs_dict, outputs_dict, Train_dict)

    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Get Paths: ###
        global_step_string = string_rjust(self.Train_dict.global_step, 6)
        index_string = string_rjust(index, 6)
        self.final_path_movie = os.path.join(self.movie_path, global_step_string, 'Movie_Results')
        self.final_inputs_original_path = os.path.join(self.inputs_original_path, global_step_string, index_string)
        self.final_inputs_noisy_path = os.path.join(self.inputs_noisy_path, global_step_string, index_string)
        self.final_inputs_patial_blur_path = os.path.join(self.spatial_blur_path, global_step_string, index_string)
        self.final_inputs_temporal_average_path = os.path.join(self.inputs_temporal_average_path, global_step_string, index_string)
        self.final_outputs_everything_concat_path = os.path.join(self.outputs_everything_concat_path, global_step_string, index_string)
        self.final_outputs_clean_path = os.path.join(self.outputs_clean_path, global_step_string, index_string)
        self.final_statistics_path_validation = self.statistics_path
        self.final_statistics_path = self.statistics_path


        ### Run Relevant Sub-Functions: ####
        inputs_dict = EasyDict(inputs_dict)
        outputs_dict = EasyDict(outputs_dict)
        self.Train_dict = Train_dict
        inputs_dict.no_GT = self.Train_dict.no_GT
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        # self.save_everything_concatenated(inputs_dict, outputs_dict, Train_dict, index)
        self.save_everything_concatenated_multiple_frames(inputs_dict, outputs_dict, Train_dict, index)
        today_date = datetime.today().strftime('%Y-%m-%d')
        self.metrics_path = os.path.join('/raid/metrics', today_date)
        self.plots_path = os.path.join('/raid/plots', today_date)
        os.makedirs(self.metrics_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        if not inputs_dict.no_GT:
            self.save_metrics(inputs_dict, outputs_dict, index)

            if index == 2: # change to n-1 for n validation images or generalize it
                avg_psnr_noisy = self.original_noisy_average_metrics_dict.PSNR
                avg_ssim_noisy = self.original_noisy_average_metrics_dict.SSIM
                avg_psnr_clean = self.original_cleaned_average_metrics_dict.PSNR
                avg_ssim_clean = self.original_cleaned_average_metrics_dict.SSIM

                self.steps.append(len(self.avg_psnr_noisy_list)*75) # change to *val_steps

                self.avg_psnr_noisy_list.append(avg_psnr_noisy)
                self.avg_ssim_noisy_list.append(avg_ssim_noisy)
                self.avg_psnr_clean_list.append(avg_psnr_clean)
                self.avg_ssim_clean_list.append(avg_ssim_clean)


                script_dir_path = os.path.join(self.metrics_path, self.movie_path.split('/')[5])
                os.makedirs(script_dir_path, exist_ok=True)
                script_plots_path = os.path.join(self.plots_path, self.movie_path.split('/')[5])
                os.makedirs(script_plots_path, exist_ok=True)

                with open(os.path.join(script_dir_path, 'noisy_metrics.txt'), 'a') as fd:
                    fd.write(f'step {self.steps[-1]}\nPSNR: {round(avg_psnr_noisy,3)} SSIM: {round(avg_ssim_noisy,3)}\n')

                with open(os.path.join(script_dir_path, 'clean_metrics.txt'), 'a') as fd:
                    fd.write(f'step {self.steps[-1]}\nPSNR: {round(avg_psnr_clean,3)} SSIM: {round(avg_ssim_clean,3)}\n')

                def save_plot_fig(y,type, metric):
                    fig = plt.figure()
                    plt.plot(self.steps, y)
                    fig.suptitle('Avg ' + type + ' ' + metric, fontsize=20)
                    plt.xlabel('Training step', fontsize=18)
                    plt.ylabel('Avg ' + metric, fontsize=18)
                    fig.savefig(os.path.join(script_plots_path, 'Avg_' + type + '_' + metric + '.png'))

                save_plot_fig(self.avg_psnr_noisy_list, 'Noisy', 'PSNR')
                save_plot_fig(self.avg_ssim_noisy_list, 'Noisy', 'SSIM')
                save_plot_fig(self.avg_psnr_clean_list, 'Clean', 'PSNR')
                save_plot_fig(self.avg_ssim_clean_list, 'Clean', 'SSIM')







        #self.get_statistics(inputs_dict, outputs_dict, Train_dict, index)   # only prints plt the statistics, leads to Huge problems
# a = torch_get_4D(self.center_frame_original_torch_stretched).clamp(0,1)
# imshow_torch(a[0])
# plt.show(block=True)
    def save_inputs(self, inputs_dict, index):
        ### Save Simple Uint8 Format: ###
        #(1). Center Frame Original:
        save_image_torch(self.final_inputs_original_path, string_rjust(index, 6) + '_center_frame_original.png',
                         torch_tensor=torch_get_4D(self.center_frame_original_torch_stretched).clamp(0,1),
                         flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
        #(2). Center Frame Noisy:
        save_image_torch(self.final_inputs_noisy_path, string_rjust(index, 6) + '_center_frame_noisy.png',
                         torch_tensor=torch_get_4D(self.center_frame_noisy_torch_stretched).clamp(0,1),
                         flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        # ### Save Spatial Averaging Result: ###
        # blur_layer = Gaussian_Blur_Layer(1, 25, 3)
        # save_image_torch(self.final_inputs_patial_blur_path, string_rjust(index, 6) + '_center_frame_noisy_GaussianBlur.png',
        #                  torch_tensor=blur_layer(torch_get_4D(self.center_frame_noisy_torch_stretched.cpu())).clamp(0,1),
        #                  flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
        #                  flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        ### Save Temporal Simple Average: ###
        save_image_torch(self.final_inputs_temporal_average_path, string_rjust(index, 6) + '_center_frame_noisy_TemporalAverage.png',
                         torch_tensor=torch_get_4D(self.center_frame_actual_mean_torch_stretched.clamp(0,1)),
                         flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

    def save_everything_concatenated(self, inputs_dict, outputs_dict, Train_dict, index):
        ### Save Simple Uint8 Format: ###
        concatenated_tensor = torch.cat((torch_get_4D(self.center_frame_original_torch_stretched),
                                         torch_get_4D(self.center_frame_noisy_torch_stretched),
                                         torch_get_4D(self.clean_frame_estimate_torch_stretched)), -1)
        save_image_torch(self.final_outputs_everything_concat_path, string_rjust(index, 6) + '_everything_concat.png',
                         torch_tensor=concatenated_tensor.clamp(0, 1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

    def save_everything_concatenated_multiple_frames(self, inputs_dict, outputs_dict, Train_dict, index):
        ### Save Simple Uint8 Format: ###
        T = self.model_output.shape[1]
        print('Saving at: ' + self.final_outputs_everything_concat_path)
        for frame_index in np.arange(T):
            # self.normalization_stretch_factor = 1 / (inputs_dict['center_frame_noisy'].cpu().max() + 1e-3) * 0.9
            self.normalization_stretch_factor2 = 1 / (inputs_dict['center_frame_original'].cpu().max() + 1e-3) * 0.9
            current_original = torch_get_4D(self.output_frames_original[:,frame_index]) * self.normalization_stretch_factor2
            current_clean = torch_get_4D(self.model_output[:,frame_index]) * self.normalization_stretch_factor2

            if inputs_dict.no_GT:
                concatenated_tensor = torch.cat(( # when there is no original
                                                 current_original,
                                                 current_clean), -1)
            else:
                current_noisy = torch_get_4D(
                    self.output_frames_noisy[:, frame_index]) * self.normalization_stretch_factor
                concatenated_tensor = torch.cat((current_original,
                                                 current_noisy,
                                                 current_clean), -1)

            concatenated_tensor = concatenated_tensor.cpu().clamp(0,1).numpy()[0] * 255
            if concatenated_tensor.shape[0] == 1:
                concatenated_tensor = BW2RGB(concatenated_tensor[0])

            ### Add Text: ###
            H, W = self.model_output.shape[-2:]
            left_text_origin = (np.int(3*W*1/6), np.int(H*1/6))
            center_text_origin = (np.int(3*W*3/6), np.int(H*1/6))
            right_text_origin = (np.int(3*W*5/6), np.int(H*1/6))
            # cv2.putText(concatenated_tensor, 'GT', org=left_text_origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)
            # cv2.putText(concatenated_tensor, 'Noisy', org=center_text_origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)
            # cv2.putText(concatenated_tensor, 'Network Output', org=right_text_origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=1)
            if concatenated_tensor.shape[0] == 3:
                concatenated_tensor = concatenated_tensor.transpose((1, 2, 0))
            #(1). Save Concatenated Outputs:
            save_image_numpy(folder_path=os.path.join(self.final_outputs_everything_concat_path, 'Concatenated_Outputs'),
                             filename=string_rjust(index, 6) + '_everything_concat_Frame' + string_rjust(frame_index,3) + '.png',
                             numpy_array=concatenated_tensor,
                             flag_convert_bgr2rgb=True, flag_scale=False, flag_save_uint16=False, flag_convert_to_uint8=True)
            # save_image_torch(os.path.join(self.final_outputs_everything_concat_path, 'Concatenated_Outputs'),
            #                  string_rjust(index, 6) + '_everything_concat_Frame' + string_rjust(frame_index,3) + '.png',
            #                  torch_tensor=concatenated_tensor.clamp(0, 1),
            #                  flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
            #                  flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
            #(2). Save Only Network Outputs:
            save_image_torch(os.path.join(self.final_outputs_everything_concat_path, 'Network_Outputs'),
                             string_rjust(frame_index, 8) + '.png',
                             torch_tensor=current_clean.clamp(0, 1),
                             flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
            # (3). Save Only Noisy Frames:
            if not inputs_dict.no_GT:
                save_image_torch(os.path.join(self.final_outputs_everything_concat_path, 'Noisy'),
                                 string_rjust(frame_index, 8) + '.png',
                                 torch_tensor=current_noisy.clamp(0, 1),
                                 flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                                 flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
            # (4). Save Only Original Frames:
            save_image_torch(os.path.join(self.final_outputs_everything_concat_path, 'GT'),
                             string_rjust(frame_index, 8) + '.png',
                             torch_tensor=current_original.clamp(0, 1),
                             flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

    def save_outputs(self, outputs, index):
        save_image_torch(self.final_outputs_clean_path, string_rjust(index, 6) + '_clean_frame_estimate.png',
                         torch_tensor=torch_get_4D(self.clean_frame_estimate_torch_stretched.clamp(0,1)),
                         flag_convert_bgr2rgb=True, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)


    def get_statistics_validation(self, inputs_dict, outputs_dict, Train_dict, index):
        #TODO: the condition here is relevant for validation, i can unify them two using an if statement
        if index == Train_dict.test_dataset_length-1:
            # final_path = os.path.join(self.statistics_path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
            # path_make_path_if_none_exists(os.path.join(final_path, 'Results'))
            for key in self.original_noisy_history_metrics_dict.keys():
                try:
                    y1 = np.array(self.original_noisy_history_metrics_dict.inner_dict[key])
                    y2 = np.array(self.original_cleaned_history_metrics_dict.inner_dict[key])
                    # y3 = np.array(self.original_running_mean_history_metrics_dict.inner_dict[key])
                    plot_multiple2([y1, y2],
                                  legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                                 'original-clean: ' + decimal_notation(y2.mean(), 2)],
                                  super_title=key + ' over time', x_label='frame-counter', y_label=key)
                    plt.savefig(os.path.join(self.final_statistics_path_validation, 'Results', key + ' over time.png'))  #here the statistics is over the entire validation
                    plt.close('all')
                except:
                    plt.close('all')


    def get_statistics(self, inputs_dict, outputs_dict, Train_dict, index):
        path_make_path_if_none_exists(os.path.join(self.statistics_path,'Results'))
        for key in self.original_noisy_history_metrics_dict.keys():
            try:
                y1 = np.array(self.original_noisy_history_metrics_dict.inner_dict[key])
                y2 = np.array(self.original_cleaned_history_metrics_dict.inner_dict[key])
                plot_multiple([y1,y2],
                              legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                             'original-cleaned: ' + decimal_notation(y2.mean(), 2)],
                              super_title=key + ' over time', x_label='frame-counter', y_label=key)
                plt.savefig(os.path.join(self.final_statistics_path, 'Results', key + ' over time.png'))
                plt.close('all')
            except:
                plt.close('all')

    def get_statistics_running_mean(self, inputs_dict, outputs_dict, Train_dict, index):
        if index == Train_dict.test_dataset_length-1:
            path_make_path_if_none_exists(os.path.join(self.final_statistics_path_validation, 'Results'))
            for key in self.original_noisy_history_metrics_dict.keys():
                try:
                    y1 = np.array(self.original_noisy_history_metrics_dict.inner_dict[key])
                    y2 = np.array(self.original_cleaned_history_metrics_dict.inner_dict[key])
                    y3 = np.array(self.original_running_mean_history_metrics_dict.inner_dict[key])
                    plot_multiple2([y1, y2, y3],
                                  legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                                 'cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
                                                 'running mean-noisy: ' + decimal_notation(y3.mean(), 2)],
                                  super_title=key + ' over time', x_label='frame-counter', y_label=key)
                    plt.savefig(os.path.join(self.final_statistics_path_validation, 'Results', key + ' over time.png'))
                    plt.close('all')
                except:
                    plt.close('all')

    def save_metrics(self, inputs_dict, outputs, index):
        ### Get Metrics: ###
        original_noisy_metrics_dict = get_metrics_image_pair_torch(self.center_frame_noisy, self.center_frame_original)
        original_clean_estimate_metrics_dict = get_metrics_image_pair_torch(self.clean_frame_estimate, self.center_frame_original)

        ### Update Internal Dictionaries (History & Averages Dicts): ###
        self.original_noisy_history_metrics_dict.update_dict(original_noisy_metrics_dict)
        self.original_cleaned_history_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
        self.original_noisy_average_metrics_dict.update_dict(original_noisy_metrics_dict)
        self.original_cleaned_average_metrics_dict.update_dict(original_clean_estimate_metrics_dict)

        return original_noisy_metrics_dict




    def save_metrics_running_mean(self, inputs_dict, outputs, index):
        ### Get Metrics: ###
        original_noisy_metrics_dict = get_metrics_image_pair_torch(self.center_frame_noisy, self.center_frame_original)
        original_clean_estimate_metrics_dict = get_metrics_image_pair_torch(self.clean_frame_estimate, self.center_frame_original)
        original_running_mean_estimate_metrics_dict = get_metrics_image_pair_torch(self.center_frame_pseudo_running_mean, self.center_frame_original)
        # get_metrics_video_lists()

        ### Update Internal Dictionaries (History & Averages Dicts): ###
        self.original_noisy_history_metrics_dict.update_dict(original_noisy_metrics_dict)
        self.original_cleaned_history_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
        self.original_running_mean_history_metrics_dict.update_dict(original_running_mean_estimate_metrics_dict)
        self.original_noisy_average_metrics_dict.update_dict(original_noisy_metrics_dict)
        self.original_cleaned_average_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
        self.original_running_mean_average_metrics_dict.update_dict(original_running_mean_estimate_metrics_dict)

    def save_movie(self, inputs_dict, outputs_dict, index):
        if self.Video_Writer_Original_Noisy is None:
            path_make_path_if_none_exists(self.final_path_movie)
            final_movie_name1 = os.path.join(self.final_path_movie, 'Original_Noisy.avi')
            final_movie_name2 = os.path.join(self.final_path_movie, 'Original_Cleaned.avi')
            final_movie_name3 = os.path.join(self.final_path_movie, 'Noisy_Cleaned.avi')
            final_movie_name4 = os.path.join(self.final_path_movie, 'Noisy.avi')
            final_movie_name5 = os.path.join(self.final_path_movie, 'Cleaned.avi')
            final_movie_name6 = os.path.join(self.final_path_movie, 'Original.avi')
            final_movie_name7 = os.path.join(self.final_path_movie, 'Noisy_Cleaned_Original.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # Be sure to use lower case
            center_frame_original = inputs_dict['center_frame_original']
            B,C,H,W = center_frame_original.shape
            self.Video_Writer_Original_Noisy = cv2.VideoWriter(final_movie_name1, fourcc, 25.0, (2*W, H))
            self.Video_Writer_Original_Cleaned = cv2.VideoWriter(final_movie_name2, fourcc, 25.0, (2*W, H))
            self.Video_Writer_Noisy_Cleaned = cv2.VideoWriter(final_movie_name3, fourcc, 25.0, (2*W, H))
            self.Video_Writer_Noisy_Cleaned_Original = cv2.VideoWriter(final_movie_name7, fourcc, 25.0, (3*W, H))
            self.Video_Writer_Noisy = cv2.VideoWriter(final_movie_name4, fourcc, 25.0, (W, H))
            self.Video_Writer_Cleaned = cv2.VideoWriter(final_movie_name5, fourcc, 25.0, (W, H))
            self.Video_Writer_Original = cv2.VideoWriter(final_movie_name6, fourcc, 25.0, (W, H))

        ### Get Numpy Frames: ###
        clean_estimate_image = self.clean_frame_estimate_numpy_stretched.clip(0,1)
        center_frame_original = self.center_frame_original_numpy_stretched.clip(0,1)
        center_frame_noisy = self.center_frame_noisy_numpy_stretched.clip(0,1)
        clean_estimate_image = torch_get_3D(clean_estimate_image)
        center_frame_original = torch_get_3D(center_frame_original)
        center_frame_noisy = torch_get_3D(center_frame_noisy)

        original_noisy_concat_frame = np.concatenate([center_frame_original,center_frame_noisy],1)
        original_cleaned_concat_frame = np.concatenate([center_frame_original,clean_estimate_image],1)
        noisy_cleaned_concat_frame = np.concatenate([center_frame_noisy,clean_estimate_image],1)
        noisy_cleaned_original_concat_frame = np.concatenate([center_frame_noisy,clean_estimate_image,center_frame_original],1)

        ### BW to "Pseudo RGB": ###
        original_noisy_concat_frame = numpy_array_to_video_ready(original_noisy_concat_frame)
        original_cleaned_concat_frame = numpy_array_to_video_ready(original_cleaned_concat_frame)
        noisy_cleaned_concat_frame = numpy_array_to_video_ready(noisy_cleaned_concat_frame)
        noisy_cleaned_original_concat_frame = numpy_array_to_video_ready(noisy_cleaned_original_concat_frame)
        clean_estimate_image = numpy_array_to_video_ready(clean_estimate_image)
        center_frame_original = numpy_array_to_video_ready(center_frame_original)
        center_frame_noisy = numpy_array_to_video_ready(center_frame_noisy)

        ### Write Frames To Movie: ###
        self.Video_Writer_Original_Noisy.write(original_noisy_concat_frame)
        self.Video_Writer_Original_Cleaned.write(original_cleaned_concat_frame)
        self.Video_Writer_Noisy_Cleaned_Original.write(noisy_cleaned_original_concat_frame)
        self.Video_Writer_Noisy_Cleaned.write(noisy_cleaned_concat_frame)
        self.Video_Writer_Noisy.write(center_frame_noisy)
        self.Video_Writer_Cleaned.write(clean_estimate_image)
        self.Video_Writer_Original.write(center_frame_original)

        if index == self.Train_dict.num_mini_batches_val-1:
            self.Video_Writer_Original_Noisy.release()
            self.Video_Writer_Noisy_Cleaned_Original.release()
            self.Video_Writer_Original_Cleaned.release()
            self.Video_Writer_Noisy_Cleaned.release()
            self.Video_Writer_Noisy.release()
            self.Video_Writer_Cleaned.release()
            self.Video_Writer_Original.release()

class InferenceCallback_Denoising_MultipleOutputFrames_Base(GeneralCallback):
    def __init__(self, output_path, Train_dict=None):
        super().__init__(output_path, Train_dict)
        self.is_training = Train_dict.is_training

    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Get Paths: ###
        global_step_string = string_rjust(self.Train_dict.global_step, 6)
        index_string = string_rjust(index, 6)
        self.final_path_movie = os.path.join(self.movie_path, global_step_string, 'Movie_Results')
        self.final_inputs_original_path = os.path.join(self.inputs_original_path, global_step_string, index_string)
        self.final_inputs_noisy_path = os.path.join(self.inputs_noisy_path, global_step_string, index_string)
        self.final_inputs_patial_blur_path = os.path.join(self.spatial_blur_path, global_step_string, index_string)
        self.final_inputs_temporal_average_path = os.path.join(self.inputs_temporal_average_path, global_step_string,
                                                               index_string)
        self.final_outputs_everything_concat_path = os.path.join(self.outputs_everything_concat_path,
                                                                 global_step_string, index_string)
        self.final_outputs_clean_path = os.path.join(self.outputs_clean_path, global_step_string, index_string)
        self.final_statistics_path_validation = self.statistics_path
        self.final_statistics_path = self.statistics_path

        ### Run Relevant Sub-Functions: ####
        inputs_dict = EasyDict(inputs_dict)
        outputs_dict = EasyDict(outputs_dict)
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_everything_concatenated(inputs_dict, outputs_dict, Train_dict, index)
        self.save_everything_concatenated_multiple_frames(inputs_dict, outputs_dict, Train_dict, index)
        self.save_metrics(inputs_dict, outputs_dict, index)
        # self.get_statistics(inputs_dict, outputs_dict, Train_dict, index)   # only prints plt the statistics, leads to Huge problems


from RapidBase.Utils.Registration.Warp_Layers import Warp_Object, Warp_Tensors_Affine_Layer
class InferenceCallback_OpticalFlow_Base(InferenceCallback_Denoising_Base):
    def __init__(self, output_path, Train_dict=None):
        super().__init__(output_path, Train_dict)
        self.upsample_layer = nn.Upsample(scale_factor=Train_dict.downsampling_factor, mode='bilinear')
        self.warp_object = Warp_Object()
        self.affine_warp_object = Warp_Tensors_Affine_Layer()

    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(inputs_dict, outputs_dict, index)

    def save_outputs(self, inputs_dict, outputs, index):
        path = self.path

        ### Assign Optical Flow Outputs To Internal Variables: ###
        self.final_optical_flow_estimate = outputs.optical_flow_estimate

        ### Shift Image According To Optical Flow / Translation (different models can have different reference indices, and some models can have multiple images): ###
        self.original_frame_warped = self.warp_object.forward(input_image=inputs_dict.output_frames_original[:,1:2],
                                                         delta_x=self.final_optical_flow_estimate[:,0:1], delta_y=self.final_optical_flow_estimate[:,1:2])
        # original_frame_shifted = self.affine_warp_object.forward(input_tensors=id.output_frames_original[:, 1:2],
        #                                                  shift_x=outputs.translation_estimate[:, 0:1],
        #                                                  shift_y=outputs.translation_estimate[:, 1:2], scale=1, rotation_angle=0)
        self.original_frame_warped_stretched = self.original_frame_warped * self.normalization_stretch_factor

        ### Save original image after warping: ###
        final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        save_image_torch(final_path, string_rjust(index, 6) + '_original_frame_warped.png',
                         torch_tensor=self.original_frame_warped_stretched.clamp(0, 1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        ### Save Optical Flow Output: ###
        #TODO: add possibility of stretching the image (imagesc) but let the colorbar tell us the values range
        #TODO: add optical flow visualization (intensity/arrows)
        self.final_optical_flow_estimate_stretched = self.final_optical_flow_estimate
        self.final_optical_flow_estimate_intensity = torch.sqrt(self.final_optical_flow_estimate[:,0:1]**2 + self.final_optical_flow_estimate[:,1:2]**2)
        save_image_torch(final_path, string_rjust(index, 6) + '_OpticalFlow_X.png',
                         torch_tensor=self.final_optical_flow_estimate_stretched[0, 0:1],
                         flag_convert_bgr2rgb=False, flag_scale_by_255=False, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False, flag_colorbar=True)
        save_image_torch(final_path, string_rjust(index, 6) + '_OpticalFlow_Y.png',
                         torch_tensor=self.final_optical_flow_estimate_stretched[0, 1:2],
                         flag_convert_bgr2rgb=False, flag_scale_by_255=False, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False,
                         flag_colorbar=True)
        save_image_torch(final_path, string_rjust(index, 6) + '_OpticalFlow_Intensity.png',
                         torch_tensor=self.final_optical_flow_estimate_intensity,
                         flag_convert_bgr2rgb=False, flag_scale_by_255=False, flag_array_to_uint8=False,
                         flag_imagesc=True, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False,
                         flag_colorbar=True)


### Normal Training: ###
from RapidBase.Utils.ML_and_Metrics.Metrics import get_metrics_image_pair_torch
from RapidBase.Utils.MISCELENEOUS import AverageMeter_Dict, AverageMeter, KeepValuesHistory_Dict
from RapidBase.Utils.MISCELENEOUS import string_rjust
from RapidBase.Utils.IO.Path_and_Reading_utils import path_make_path_if_none_exists
from RapidBase.Utils.MISCELENEOUS import decimal_notation
from RapidBase.Utils.IO.Imshow_and_Plots import plot_multiple
from RapidBase.Basic_Import_Libs import *
class InferenceCallback_ValidationDataSet(InferenceCallback_Denoising_Base):
    # Save original, noisy and clean. mainly used for validation or when i want to get statistics over a dataset in a single folder
    def __init__(self, output_path, Train_dict=None):
        super().__init__(output_path)
        self.is_training = Train_dict.is_training
        self.original_noisy_average_metrics_dict = AverageMeter_Dict()
        self.original_cleaned_average_metrics_dict = AverageMeter_Dict()
        self.original_noisy_history_metrics_dict = KeepValuesHistory_Dict()
        self.original_cleaned_history_metrics_dict = KeepValuesHistory_Dict()
        self.Train_dict = Train_dict
        pass

    def run(self, index, inputs_dict, outputs_dict, Train_dict):
        ### Run Relevant Sub-Functions: ####
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_metrics(inputs_dict, outputs_dict, index)




### For Movies: ###
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import scale_array_to_range
class InferenceCallback_Movie(InferenceCallback_Denoising_Base):
    ### Callback that only saves the clean estimate.
    # this is relevant, for instance, when creating a cleaned version of a movie and i want to create a folder with only those frames
    def __init__(self, output_path, Train_dict=None):
        super().__init__(output_path, Train_dict)
        self.is_training = Train_dict.is_training
        self.original_noisy_average_metrics_dict = AverageMeter_Dict()
        self.original_cleaned_average_metrics_dict = AverageMeter_Dict()
        self.original_noisy_history_metrics_dict = KeepValuesHistory_Dict()
        self.original_cleaned_history_metrics_dict = KeepValuesHistory_Dict()
        self.Train_dict = Train_dict
        self.Video_Writer_Original_Noisy = None

        ### Paths: ###
        self.inputs_original_path = self.path
        self.inputs_noisy_path = self.path
        self.inputs_temporal_average_path = self.path
        self.outputs_clean_path = self.path
        self.spatial_blur_path = self.path
        self.outputs_everything_concat_path = self.path
        self.statistics_path = self.path
        self.movie_path = self.path

        pass

    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Get Paths: ###
        global_step_string = string_rjust(self.Train_dict.global_step, 6)
        index_string = string_rjust(index, 6)
        self.final_path_movie = os.path.join(self.movie_path, global_step_string, 'Movie_Results')
        self.final_inputs_original_path = os.path.join(self.inputs_original_path, global_step_string , 'inputs_original')
        self.final_inputs_noisy_path = os.path.join(self.inputs_noisy_path, global_step_string, 'inputs_noisy')
        self.final_inputs_patial_blur_path = os.path.join(self.spatial_blur_path, global_step_string, 'inputs_spatial_blur')
        self.final_inputs_temporal_average_path = os.path.join(self.inputs_temporal_average_path, global_step_string, 'inputs_temporal_blur')
        self.final_outputs_everything_concat_path = os.path.join(self.outputs_everything_concat_path, global_step_string, 'everything_concat')
        self.final_outputs_clean_path = os.path.join(self.outputs_clean_path, global_step_string, 'outputs_clean')
        self.final_statistics_path_validation = os.path.join(self.statistics_path, global_step_string, 'validation_statistics')
        self.final_statistics_path = os.path.join(self.statistics_path, global_step_string, 'train_statistics')

        path_make_path_if_none_exists(self.final_path_movie)
        path_make_path_if_none_exists(self.final_inputs_original_path)
        path_make_path_if_none_exists(self.final_inputs_noisy_path)
        path_make_path_if_none_exists(self.final_inputs_patial_blur_path)
        path_make_path_if_none_exists(self.final_inputs_temporal_average_path)
        path_make_path_if_none_exists(self.final_outputs_everything_concat_path)
        path_make_path_if_none_exists(self.final_outputs_clean_path)
        path_make_path_if_none_exists(self.final_statistics_path_validation)
        path_make_path_if_none_exists(self.final_statistics_path)

        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_everything_concatenated(inputs_dict, outputs_dict, Train_dict, index)
        # self.save_metrics(inputs_dict, outputs_dict, index)
        # self.get_statistics_validation(inputs_dict, outputs_dict, Train_dict, index)
        self.save_movie(inputs_dict, outputs_dict, index)



class InferenceCallback_Movie_ExtraDataset(GeneralCallback):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        inputs_dict['center_frame_original'] = Train_dict.original_images_test_dataset[index]['center_frame_original'].unsqueeze(0)
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_metrics(inputs_dict, outputs_dict, index)
        self.get_statistics(inputs_dict, outputs_dict, Train_dict, index)
        self.save_movie(inputs_dict, outputs_dict, index)



from RapidBase.Utils.IO.Imshow_and_Plots import plot_multiple as plot_multiple2
from RapidBase.Utils.Tensor_Manipulation.Pytorch_Numpy_Utils import Gaussian_Blur_Layer
class InferenceCallback_Denoise(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_metrics(inputs_dict, outputs_dict, index)
        # self.get_statistics(inputs_dict, outputs_dict, Train_dict, index)


class InferenceCallback_Denoise_OnlyStats(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.save_metrics(inputs_dict, outputs_dict, index)


class InferenceCallback_Denoise_ExternalOriginal(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        inputs_dict['center_frame_original'] = Train_dict.original_images_test_dataset[index]['center_frame_original'].unsqueeze(0)
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_metrics(inputs_dict, outputs_dict, index)
        self.get_statistics(inputs_dict, outputs_dict, Train_dict, index)


class InferenceCallback_Denoise_WithRunningMean(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_metrics_running_mean(inputs_dict, outputs_dict, index)
        self.get_statistics_running_mean(inputs_dict, outputs_dict, Train_dict, index)


from RapidBase.Utils.IO.Imshow_and_Plots import plot_torch
import matplotlib.pyplot as plt
class InferenceCallback_Denoising_Recursive_Base(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.upsample_layer = nn.Upsample(scale_factor=Train_dict.downsampling_factor, mode='bilinear')
        self.downsampling_factor = Train_dict.downsampling_factor
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs_recursive(inputs_dict, outputs_dict, index)
        self.save_metrics(inputs_dict, outputs_dict, index)
        self.get_statistics_recursive_running_mean_running_mean(inputs_dict, outputs_dict, Train_dict, index)

    
    def clean_dicts(self, inputs_dict):
        self.original_noisy_average_metrics_dict = AverageMeter_Dict()
        self.original_cleaned_average_metrics_dict = AverageMeter_Dict()
        self.original_running_mean_average_metrics_dict = AverageMeter_Dict()
        self.original_noisy_history_metrics_dict = KeepValuesHistory_Dict()
        self.original_cleaned_history_metrics_dict = KeepValuesHistory_Dict()
        self.original_running_mean_history_metrics_dict = KeepValuesHistory_Dict()


    def get_statistics_recursive_running_mean_running_mean(self, inputs_dict, outputs_dict, Train_dict, index):
        flag_finished_going_over_validation_set = (index == Train_dict.test_dataset_length-1 and Train_dict.now_training == False)
        flag_debugging_during_training = Train_dict.now_training
        if (flag_finished_going_over_validation_set) or (flag_debugging_during_training):
            path = self.path
            final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
            path_make_path_if_none_exists(os.path.join(final_path, 'Results'))
            for key in self.original_noisy_history_metrics_dict.keys():
                try:
                    y1 = np.array(self.original_noisy_history_metrics_dict.inner_dict[key])
                    y2 = np.array(self.original_cleaned_history_metrics_dict.inner_dict[key])
                    y3 = np.array(self.original_running_mean_history_metrics_dict.inner_dict[key])
                    plot_multiple2([y1, y2, y3],
                                  legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                                 'cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
                                                 'running mean-noisy: ' + decimal_notation(y3.mean(), 2)],
                                  super_title=key + ' over time', x_label='frame-counter', y_label=key)
                    plt.savefig(os.path.join(final_path, 'Results', key + ' over time.png'))
                    plt.close('all')
                except:
                    plt.close('all')


    def save_outputs_recursive(self, inputs_dict, outputs, index):
        path = self.path
        final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))

        # ### Plot Metrics Over Time: ###
        # plt.plot(np.array(outputs.l1_losses_over_time))

        ### Loop Over Clean Frames And Save Them For Later Review: ###
        output_frames_over_time = outputs.output_frames_over_time
        for frame_counter, output_frame in enumerate(output_frames_over_time):
            ### Save Output Frames: ###
            save_image_torch(final_path, string_rjust(index, 6) + '_clean_frame_estimate_Frame' + string_rjust(frame_counter,2) + '.png',
                             torch_tensor=(self.normalization_stretch_factor*output_frame[0]).clamp(0,1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

            ### Save Output Frames: ###
            if len(inputs_dict.output_frames_noisy.shape) == 5:
                current_noisy = self.output_frames_noisy[0,frame_counter+self.frame_index_to_predict,:,:].cpu()
                current_original = self.output_frames_original[0,frame_counter+self.frame_index_to_predict,:,:].cpu()
                current_clean = output_frame[0].cpu()
            elif len(inputs_dict.output_frames_noisy.shape) == 4:
                current_noisy = self.output_frames_noisy[0,frame_counter+self.frame_index_to_predict:frame_counter+self.frame_index_to_predict+1,:,:].cpu()
                current_original = self.output_frames_original[0,frame_counter+self.frame_index_to_predict:frame_counter+self.frame_index_to_predict+1,:,:].cpu()
                current_clean = output_frame[0].cpu()
            current_concat = torch.cat((current_noisy, current_clean, current_original), -1)
            save_image_torch(os.path.join(final_path, 'concat'),
                             string_rjust(index, 6) + '_ConcatFrame' + string_rjust(frame_counter, 2) + '.png',
                             torch_tensor=(self.normalization_stretch_factor * current_concat).clamp(0, 1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

    def save_outputs_recursive_running_mean(self, inputs_dict, outputs, index):
        path = self.path
        final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))

        # ### Plot Metrics Over Time: ###
        # plt.plot(np.array(outputs.l1_losses_over_time))

        ### Loop Over Clean Frames And Save Them For Later Review: ###
        output_frames_over_time = outputs.output_frames_over_time
        for frame_counter, output_frame in enumerate(output_frames_over_time):
            ### Save Output Frames: ###
            save_image_torch(final_path, string_rjust(index, 6) + '_clean_frame_estimate_Frame' + string_rjust(frame_counter,2) + '.png',
                             torch_tensor=(self.normalization_stretch_factor*output_frame[0]).clamp(0,1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

            ### Save Input/Output/Original Frames: ###
            if len(inputs_dict.output_frames_noisy.shape) == 5:
                current_noisy = self.output_frames_noisy[0, frame_counter:frame_counter + 1, :, :].cpu()
                current_original = self.output_frames_original[0, frame_counter:frame_counter + 1, :, :].cpu()
                current_clean = output_frame.cpu()
            elif len(inputs_dict.output_frames_noisy.shape) == 4:
                current_noisy = self.output_frames_noisy[0, frame_counter:frame_counter + 1, :, :].cpu()
                current_original = self.output_frames_original[0, frame_counter:frame_counter + 1, :, :].cpu()
                current_clean = output_frame[0].cpu()
            current_approximate_running_average = self.output_frames_noisy_running_average[0,frame_counter:frame_counter+1,:,:].cpu()
            #(1). Noisy-Clean-Original:
            current_concat = torch.cat((current_noisy, current_clean, current_original), -1)
            save_image_torch(os.path.join(final_path, 'concat_NoisyCleanOriginal'),
                             string_rjust(index, 6) + '_ConcatFrame' + string_rjust(frame_counter, 2) + '.png',
                             torch_tensor=(self.normalization_stretch_factor * current_concat).clamp(0, 1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
            #(2). RunningAverage-Clean-Original:
            current_concat = torch.cat((current_approximate_running_average, current_clean, current_original), -1)
            save_image_torch(os.path.join(final_path, 'concat_RA'),
                             string_rjust(index, 6) + '_ConcatFrame' + string_rjust(frame_counter, 2) + '.png',
                             torch_tensor=(self.normalization_stretch_factor * current_concat).clamp(0, 1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

            ### Save Metrics: ###   #TODO: add Edges metrics over time
            original_noisy_metrics_dict = get_metrics_image_pair_torch(current_noisy, current_original)
            original_clean_estimate_metrics_dict = get_metrics_image_pair_torch(current_clean, current_original)
            original_running_mean_estimate_metrics_dict = get_metrics_image_pair_torch(current_approximate_running_average, current_original)

            ### Update Internal Dictionaries (History & Averages Dicts): ###
            self.original_noisy_history_metrics_dict.update_dict(original_noisy_metrics_dict)
            self.original_cleaned_history_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
            self.original_running_mean_history_metrics_dict.update_dict(original_running_mean_estimate_metrics_dict)
            self.original_noisy_average_metrics_dict.update_dict(original_noisy_metrics_dict)
            self.original_cleaned_average_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
            self.original_running_mean_average_metrics_dict.update_dict(original_running_mean_estimate_metrics_dict)

    def save_inputs(self, inputs_dict, index):
        path = self.path
        final_path1 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        final_path2 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))

        ### Save Simple Uint8 Format: ###
        #(1). Current GT Frame:
        save_image_torch(final_path1, string_rjust(index, 6) + '_center_frame_original.png',
                         torch_tensor=self.current_gt_clean_frame_torch_stretched.clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
        #(2). Current Noisy Frame:
        save_image_torch(final_path2, string_rjust(index, 6) + '_center_frame_noisy.png',
                         torch_tensor=self.current_noisy_frame_torch_stretched.clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
        #(3). Running Mean
        final_path3 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        save_image_torch(final_path3, string_rjust(index, 6) + '_center_frame_pseudo_running_mean_noisy_scaled.png',
                         torch_tensor=self.current_moving_average_torch_stretched.clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)


class DebugCallback_Denoise_Recursive(InferenceCallback_Denoising_Recursive_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.clean_dicts(inputs_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs_recursive_running_mean(inputs_dict, outputs_dict, index)
        self.get_statistics_recursive_running_mean(inputs_dict, outputs_dict, Train_dict, index)


class InferenceCallback_Denoise_WithActualMean(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        self.save_metrics_running_mean(inputs_dict, outputs_dict, index)
        self.get_statistics_running_mean(inputs_dict, outputs_dict, Train_dict, index)
        
from RapidBase.Utils.ML_and_Metrics.Metrics import get_metrics_image_pair_torch, get_metrics_image_pair
class InferenceCallback_Denoise_Recursive_ClassicIIR(InferenceCallback_Denoising_Base):
    def run(self, index, inputs_dict, outputs_dict, Train_dict=None):
        ### Run Relevant Sub-Functions: ####
        self.Train_dict = Train_dict
        self.set_internal_variables(index, inputs_dict, outputs_dict, Train_dict)
        self.save_inputs(inputs_dict, index)
        self.save_outputs(outputs_dict, index)
        if self.Train_dict.now_training:
            self.save_metrics_single_whole_scene_recursive(inputs_dict, outputs_dict, index)
        else:
            #TODO: i don't understand why the test/validation time functions are different, need to remember
            #TODO: perhapse it's because when i'm not training i'm using a different type of dataset? perhapse the validation loop is different?
            self.save_metrics(inputs_dict, outputs_dict, index)
            self.get_statistics(inputs_dict, outputs_dict, Train_dict, index) #TODO: perhapse add condition here instead of inside the function?

    def get_statistics(self, inputs_dict, outputs_dict, Train_dict, index):
        if index == Train_dict.test_dataset_length-1: #Only present statistics at the end of the test set
            path = self.path
            path_make_path_if_none_exists(os.path.join(path, 'Results'))
            for key in self.original_noisy_history_metrics_dict.keys():
                try:
                    y1 = np.array(self.original_noisy_history_metrics_dict.inner_dict[key])
                    y2 = np.array(self.original_cleaned_history_metrics_dict.inner_dict[key])
                    y3 = np.array(self.original_running_mean_history_metrics_dict.inner_dict[key])
                    plot_multiple2([y1, y2, y3],
                                  legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                                 'cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
                                                 'running mean-noisy: ' + decimal_notation(y3.mean(), 2)],
                                  super_title=key + ' over time', x_label='frame-counter', y_label=key)
                    plt.savefig(os.path.join(path, 'Results', key + ' over time.png'))
                    plt.close('all')
                except:
                    plt.close('all')

    def save_metrics_single_whole_scene_recursive(self, inputs_dict, outputs, index):        
        ### Get List Of Image Frames Over Time: ###
        original_frames_list = []
        noisy_frames_list = []
        running_average_list = []
        clean_estimate_list = []
        for frame_index in np.arange(len(outputs['output_frames_over_time'])):
            #print(frame_index)
            original_frames_list.append(inputs_dict['output_frames_original'][0, frame_index:frame_index + 1, :, :].squeeze().cpu().numpy())
            noisy_frames_list.append(inputs_dict['output_frames_noisy'][0, frame_index:frame_index + 1, :, :].squeeze().cpu().numpy())
            running_average_list.append(inputs_dict['output_frames_noisy_running_average'][0, frame_index:frame_index + 1, :, :].squeeze().cpu().numpy())
            clean_estimate_list.append(outputs['output_frames_over_time'][frame_index][0].squeeze().cpu().numpy())

        ### Get Metrics Over Lists: ###
        original_noisy_outputs_dict_average, original_noisy_outputs_dict_history = \
            get_metrics_video_lists(original_frames_list, noisy_frames_list, number_of_images=np.inf)
        original_clean_outputs_dict_average, original_clean_outputs_dict_history = \
            get_metrics_video_lists(original_frames_list, clean_estimate_list, number_of_images=np.inf)
        original_RA_outputs_dict_average, original_RA_outputs_dict_history = \
            get_metrics_video_lists(original_frames_list, running_average_list, number_of_images=np.inf)

        ### Plot Metrics Over Time: ###
        path = self.path
        final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        path_make_path_if_none_exists(os.path.join(final_path, 'Results'))
        for key in original_noisy_outputs_dict_history.keys():
            try:
                y1 = np.array(original_noisy_outputs_dict_history.inner_dict[key])
                y2 = np.array(original_clean_outputs_dict_history.inner_dict[key])
                y3 = np.array(original_RA_outputs_dict_history.inner_dict[key])
                plot_multiple([y1, y2, y3],
                               legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                              'cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
                                              'running mean-noisy: ' + decimal_notation(y3.mean(), 2)],
                               super_title=key + ' over time', x_label='frame-counter', y_label=key)
                plt.savefig(os.path.join(final_path, 'Results', key + ' over time.png'))
                plt.close('all')
            except:
                plt.close('all')


    def save_metrics(self, inputs_dict, outputs, index):
        ### Get All Relevant Images: ###
        #TODO: i think i got rid of these frm the pre-processing because they we're confusing, put them into postprocessing. 
        clean_estimate_image = outputs['clean_frame_estimate']
        center_frame_original = inputs_dict['current_gt_clean_frame']
        center_frame_noisy = inputs_dict['current_noisy_frame']
        center_frame_noisy_running_mean = inputs_dict['current_moving_average']

        ### Get Metrics: ###
        original_noisy_metrics_dict = get_metrics_image_pair_torch(center_frame_noisy, center_frame_original)
        original_clean_estimate_metrics_dict = get_metrics_image_pair_torch(clean_estimate_image, center_frame_original)
        original_running_mean_estimate_metrics_dict = get_metrics_image_pair_torch(center_frame_noisy_running_mean, center_frame_original)

        ### Update Internal Dictionaries (History & Averages Dicts): ###
        self.original_noisy_history_metrics_dict.update_dict(original_noisy_metrics_dict)
        self.original_cleaned_history_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
        self.original_running_mean_history_metrics_dict.update_dict(original_running_mean_estimate_metrics_dict)
        self.original_noisy_average_metrics_dict.update_dict(original_noisy_metrics_dict)
        self.original_cleaned_average_metrics_dict.update_dict(original_clean_estimate_metrics_dict)
        self.original_running_mean_average_metrics_dict.update_dict(original_running_mean_estimate_metrics_dict)

    def save_outputs(self, outputs, index):
        path = self.path
        final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))

        ### Loop Over Clean Frames And Save Them For Later Review: ###
        output_frames_over_time = outputs.output_frames_over_time
        for frame_counter, output_frame in enumerate(output_frames_over_time):
            ### Save Ouptut Images: ###
            save_image_torch(final_path, string_rjust(index, 6) + '_clean_frame_estimate_Frame' + string_rjust(frame_counter,4) + '.png',
                             torch_tensor=output_frame[0].clamp(0,1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

            ### Save unclipped, float32 frames for future examination: ###
            clean_frame_estimate = np.uint16(output_frame[0].cpu().detach().numpy().transpose([1, 2, 0]))
            scipy.io.savemat(os.path.join(final_path, string_rjust(index, 6) + '_clean_frame_estimate_Frame'),
                             mdict={'clean_frame_estimate': clean_frame_estimate})

            ### Save Reset Gates: ###
            save_image_torch(final_path, string_rjust(index, 6) + '_Reset_Gate_Classic' + string_rjust(frame_counter, 4) + '.png',
                             torch_tensor=outputs.reset_gates_classic_list[frame_counter][0].clamp(0, 1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
            save_image_torch(final_path, string_rjust(index, 6) + '_Reset_Gate_Final' + string_rjust(frame_counter, 4) + '.png',
                             torch_tensor=outputs.Reset_Gates_Combine_list[frame_counter][0].clamp(0, 1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

    def save_inputs(self, inputs_dict, index):
        # Disparity values are scaled by 32
        path = self.path
        final_path = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        final_path1 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        final_path3 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))

        ### Save Input Frames: ###
        for frame_counter in np.arange(inputs_dict['output_frames_original'].shape[1]):
            ### Save Ouptut Images: ###
            save_image_torch(final_path, string_rjust(index, 6) + '_Original_Frame' + string_rjust(frame_counter,4) + '.png',
                             torch_tensor=inputs_dict['output_frames_original'][0,frame_counter:frame_counter+1,:,:].clamp(0,1),
                             flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                             flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)


        center_frame_original = inputs_dict['current_gt_clean_frame'][0].cpu().detach().numpy().transpose([1, 2, 0])
        center_frame_noisy = inputs_dict['current_noisy_frame'][0].cpu().detach().numpy().transpose([1, 2, 0])
        center_frame_noisy_running_mean = inputs_dict['current_moving_average'][0].cpu().detach().numpy().transpose([1, 2, 0])

        ### Save Simple Uint8 Format: ###
        save_image_torch(final_path1, string_rjust(index, 6) + '_center_frame_original.png',
                         torch_tensor=inputs_dict['current_gt_clean_frame'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)
        save_image_torch(final_path2, string_rjust(index, 6) + '_center_frame_noisy.png',
                         torch_tensor=inputs_dict['current_noisy_frame'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        ### Save Images In Imagesc Uint8 Format To Actually See Something: ###
        final_path3 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        save_image_torch(final_path3, string_rjust(index, 6) + '_center_frame_noisy_scaled.png',
                         torch_tensor=inputs_dict['current_noisy_frame'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        ### Save Running-Mean Images In Imagesc Uint8 Format To Actually See Something: ###
        final_path3 = os.path.join(path, string_rjust(self.Train_dict.global_step, 6), string_rjust(index, 6))
        save_image_torch(final_path3, string_rjust(index, 6) + '_center_frame_pseudo_running_mean_noisy_scaled.png',
                         torch_tensor=inputs_dict['current_moving_average'][0].clamp(0,1),
                         flag_convert_bgr2rgb=False, flag_scale_by_255=True, flag_array_to_uint8=True,
                         flag_imagesc=False, flag_convert_grayscale_to_heatmap=False, flag_save_figure=False)

        ### Save unclipped, float32 frames for future examination: ###
        scipy.io.savemat(os.path.join(final_path1, string_rjust(index, 6) + '_center_frame_original.mat'), mdict={
            'center_frame_original':center_frame_original})
        scipy.io.savemat(os.path.join(final_path1, string_rjust(index, 6) + '_center_frame_noisy.mat'),
                         mdict={'center_frame_noisy': center_frame_noisy})
        scipy.io.savemat(os.path.join(final_path1, string_rjust(index, 6) + 'center_frame_noisy_running_mean.mat'),
                         mdict={'center_frame_noisy_running_mean': center_frame_noisy_running_mean})

    def rgb2bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class GradFlowPlot_Callback(GeneralCallback):
    def __init__(self, output_path, is_training=True):
        super().__init__(output_path)
        self.sigmoid_offset = 4
        self.is_training = is_training
        self.output_path = output_path

    def run(self, index, inputs_dict, outputs_dict):

        valid_mask = (inputs_dict['left_disparity_GT'][0] < 126)
        if valid_mask.sum() > 0:
            if not self.is_training:
                path = os.path.join(self.output_path, 'Inference\Image_{}'.format(index))
                if(not os.path.isdir(path)):
                    os.makedirs(path)
                self.path = path
            else:
                self.path = os.path.join(self.output_path, 'dbg')
                if (not os.path.isdir(self.path)):
                    os.mkdir(self.path)

            save_path = os.path.join(self.path, string_rjust(index, 6))
            plot_grad_flow(outputs_dict.model, save_path)

            # self.save_inputs(inputs_dict, index)
            # self.save_outputs(outputs_dict, inputs_dict, index)









