import numpy as np
import torch
from easydict import EasyDict

import RapidBase.TrainingCore.training_utils as training_utils
from RapidBase.Utils.MISCELENEOUS import decimal_notation, scientific_notation

from RapidBase.import_all import *

def data_from_dataloader_to_GPU(data_from_dataloader, device):
    for k, v in data_from_dataloader.items():
        if type(data_from_dataloader[k]) == EasyDict or type(data_from_dataloader[k]) == dict:
            data_from_dataloader[k] = data_from_dataloader_to_GPU(data_from_dataloader[k], device)
        else:
            data_from_dataloader[k] = data_from_dataloader[k].to(device)
    return data_from_dataloader


class GeneralTrainer(object):
    def __init__(self,
                 model,
                 preprocessing_object,
                 postprocessing_object,
                 Network_Optimizer_object,
                 Loss_object,
                 TB_object,
                 LR_scheduler_object,
                 Clip_gradient_object,
                 train_dataloader,
                 test_dataloader,
                 debug_callback,
                 validation_callback,
                 inference_callback,
                 Train_dict):

        self.initialize_internal_variables(model,
                                          preprocessing_object,
                                          postprocessing_object,
                                          Network_Optimizer_object,
                                          Loss_object,
                                          TB_object,
                                          LR_scheduler_object,
                                          Clip_gradient_object,
                                          train_dataloader,
                                          test_dataloader,
                                          debug_callback,
                                          validation_callback,
                                          inference_callback,
                                          Train_dict)

    def initialize_internal_variables(self, model,
                                      preprocessing_object,
                                      postprocessing_object,
                                      Network_Optimizer_object,
                                      Loss_object,
                                      TB_object,
                                      LR_scheduler_object,
                                      Clip_gradient_object,
                                      train_dataloader,
                                      test_dataloader,
                                      debug_callback,
                                      validation_callback,
                                      inference_callback,
                                      Train_dict):
        ### Get Wanted Network: ###
        self.Train_dict = Train_dict
        self.Train_dict = Train_dict
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = Train_dict.device

        self.debug_callback = debug_callback
        self.validation_callback = validation_callback
        self.inference_callback = inference_callback

        self.model = model
        self.preprocessing_object = preprocessing_object
        self.postprocessing_object = postprocessing_object
        self.Loss_object = Loss_object
        self.Network_Optimizer_object = Network_Optimizer_object
        self.TB_object_train = TB_object[0]
        self.TB_object_validation = TB_object[1]
        self.LR_scheduler_object = LR_scheduler_object
        self.Clip_gradient_object = Clip_gradient_object
        self.debug_step = self.Train_dict.debug_step
        self.Network_checkpoint_folder = Train_dict.Network_checkpoint_folder
        self.outputs_dict = EasyDict()

    def initialize_Train_function(self):
        ####################################    Initialize Stuff For Training Loop:    ####################################
        ### Initialize Loop Steps: ###
        self.flag_stop_iterating = False
        self.Train_dict.global_batch_step = 0
        self.Train_dict.global_step = 0
        self.Train_dict.current_time_step = 0
        self.Train_dict.flag_first_save = True
        self.Train_dict.minimum_learning_rate_counter = 0
        self.Train_dict.Network_lr = self.Train_dict.init_learning_rate

        ### Network To Train Mode: ###
        self.model.train()

    def Optimize_and_Log(self, total_loss, inputs_dict, outputs_dict):
        ### Update Learning Rate: ###
        self.Network_Optimizer, self.Train_dict = self.LR_scheduler_object.step(
            self.Network_Optimizer_object.Network_Optimizer, total_loss, self.Train_dict)

        ### Back-Propagate: ###
        total_loss.backward()

        ### Clip Gradient Norm: ###
        self.model, self.Train_dict = self.Clip_gradient_object.clip_gradient(self.model, self.Train_dict)

        ### Update Weights: ###
        self.Network_Optimizer_object.Network_Optimizer.step()

        ### Print Network Description String: ###
        self.print_string(inputs_dict, self.outputs_dict, self.Train_dict)

        ### Empty CUDA Cache Just In Case: ###
        # torch.cuda.empty_cache()

    def Save_Network_Checkpoint(self, current_epoch, batch_index):
        if self.Train_dict.global_step % self.Train_dict.save_models_frequency == 0 or \
                (current_epoch == self.Train_dict.number_of_epochs - 1 and
                 batch_index + self.Train_dict.batch_size >= len(self.train_dataloader.train_dataset)):
            ### Save Network and Discriminator: ###
            print('Save Network and Discriminator Models:')
            print(self.model)
            print(self.Train_dict.Network_checkpoint_prefix)

            ### Decide Which Hyper-Parmaters To Save - From Project Or From Previous Checkpoint: ###
            current_Network_iteration_string, basic_Network_filename = \
                training_utils.Get_Network_Save_Basic_Names(self.Train_dict.global_step, self.Train_dict)
            self.Train_dict = \
                training_utils.Get_HyperParameters_Save_Flags(self.Train_dict.flag_first_save,
                                                              self.Train_dict)

            ### Save Dict: ###
            self.Train_dict.basic_Network_filename = basic_Network_filename
            self.Train_dict.variables_dictionary = None

            ### Save Network: ###
            training_utils.save_Network_parts_to_checkpoint(
                self.model, self.Network_checkpoint_folder,
                self.Network_Optimizer_object.Network_Optimizer,
                self.Train_dict)

            ### First Save Flag: ###
            self.Train_dict.flag_first_save = False

    def Log_to_TensorBoard(self, inputs_dict):
        if self.Train_dict.flag_write_TB and (self.Train_dict.global_step % self.Train_dict.tensorboard_train_frequency == 0):
            self.TB_object_train.get_TensorBoard_Metrics_Train(inputs_dict, self.outputs_dict)
            self.TB_object_train.Log_TensorBoard_Train(self.Train_dict, model=self.model)

    def Run_Debug_Calllback(self, inputs_dict):
        if self.debug_callback and (self.Train_dict.global_step % self.debug_step == 0):
            self.outputs_dict.model_state = self.model.state_dict()
            self.debug_callback.run(self.Train_dict.global_step, inputs_dict, self.outputs_dict, self.Train_dict)

    def inference(self):
        self.Train_dict.global_step = 0
        self._validate(self.Train_dict, self.other_device, self.test_dataloader, debug_callback=self.validation_callback)

    def Run_ValidationSet(self):
        if (self.Train_dict.global_step) % self.Train_dict.validation_set_frequency == 0 and self.Train_dict.flag_do_validation:
            print('########## Running validation set ##########')
            self._validate(self.Train_dict, self.Train_dict.device, self.test_dataloader,
                           debug_callback=self.validation_callback)
            self.model.train()

    def forward_propagation(self, Train_dict, inputs_dict):
        ### PreProcess the inputs: ###
        network_input, inputs_dict, self.Train_dict, self.model = self.preprocessing_object.forward(inputs_dict, self.Train_dict, self.model)

        ### Pass inputs to through the model: ###
        # model_output_tensor,_,_ = self.model.forward(network_input)

        model_output = self.model.forward(network_input)
        self.outputs_dict.model_output = model_output

        ### Post process network's outputs: ###
        model_output_tensor, self.outputs_dict, inputs_dict, self.Train_dict, self.model = \
            self.postprocessing_object.forward(inputs_dict, self.outputs_dict, self.Train_dict, self.model)

        ### Return: ###
        return inputs_dict, model_output, self.outputs_dict


    def print_string(self, inputs_dict, outputs_dict, Train_dict):
        #TODO: change this to an object instead of an internal function
        ### General Stats: ###
        network_name_string = 'Network Name: ' + Train_dict.Network_checkpoint_prefix
        epoch_batch_string = 'EPOCH {:05d}, BATCH {}/{}:   '.format(Train_dict.current_epoch, Train_dict.batch_index, Train_dict.num_mini_batches_trn)

        ### Print String: ###
        print_string = ''
        print_string += network_name_string + ', ' + epoch_batch_string + ', '
        print_string += 'G Step ' + scientific_notation(Train_dict.global_step) + ', '
        print_string += 'Gen lr: ' + scientific_notation(self.LR_scheduler_object.Network_lr) + ', \n'

        ### Losses (Project Specific): ###
        print_string += 'Total Loss: ' + scientific_notation(outputs_dict.total_loss.item()) + ', '

        ### Correct Training Indicators: ###
        #(*) i only search for (min,max) in first batch example to avoid too much calculation
        # network_final_output_min = outputs_dict.model_output['temporal_part_output_frame'][0].min().item()
        # network_final_output_max = outputs_dict.model_output['temporal_part_output_frame'][0].max().item()
        network_final_output_min = outputs_dict.model_output[0].min().item()
        network_final_output_max = outputs_dict.model_output[0].max().item()
        network_input_min = inputs_dict['center_frame_original'][0].min().item()
        network_input_max = inputs_dict['center_frame_original'][0].max().item()
        print_string += ' Network_Range=[' + decimal_notation(network_final_output_min) + ',' + decimal_notation(network_final_output_max) + '] , ' + \
                        ' GT_Range=[' + decimal_notation(network_input_min) + ',' + decimal_notation(network_input_max) + '] , '
        print_string += 'Grad-Norm: ' + scientific_notation(Train_dict.total_norm)
        toc(print_string)


    def Train(self):

        ### Initialize Train Function: ###
        self.initialize_Train_function()

        ### Training Loop: ###
        for current_epoch in np.arange(0, self.Train_dict.number_of_epochs):
            # tic()
            ### If we got a signal to stop iterating then break: ###
            if self.flag_stop_iterating:
                break

            ### Empty CUDA Cache Just In Case: ###
            # torch.cuda.empty_cache()

            ### Loop over the current epoch data: ###
            tic()
            for batch_index, data_from_dataloader in enumerate(self.train_dataloader):
                toc('dataloading')
                tic()
                ### If we are at the end of our epoch or we got a signal to stop iterating -> break and stop: ###
                self.flag_stop_iterating = (self.Train_dict.global_batch_step == self.Train_dict.max_total_number_of_batches or self.flag_stop_iterating == True)
                if self.flag_stop_iterating:
                    break

                if self.flag_stop_iterating == False:
                    ##########################################################
                    ### Empty CUDA Cache Just In Case: ###
                    # torch.cuda.empty_cache()

                    ##########################################################

                    ##########################################################
                    ### Run Validation Set If Wanted: ###
                    self.Run_ValidationSet()
                    ##########################################################

                    ##########################################################
                    ### Uptick Steps: ###
                    self.Train_dict.global_step += 1
                    self.Train_dict.current_epoch = current_epoch
                    self.Train_dict.batch_index = batch_index
                    self.Train_dict.global_batch_step += 1  # total number of batches so far


                    ### Data From DataLoader To GPU & To EasyDict: ###
                    if self.Train_dict.flag_all_inputs_to_gpu:
                        data_from_dataloader = data_from_dataloader_to_GPU(data_from_dataloader, self.device)
                    inputs_dict = EasyDict(data_from_dataloader)

                    ### Training Flag: ###
                    self.Train_dict.now_training = True

                    self.Network_Optimizer_object.Network_Optimizer.zero_grad()

                    ### Pre-Process + Forward Pass + Post-Process: ###
                    inputs_dict, model_output_tensor, self.outputs_dict = self.forward_propagation(self.Train_dict, inputs_dict)

                    ### Get Train Loss: ###
                    total_loss, self.outputs_dict, inputs_dict = self.Loss_object.forward(inputs_dict, self.outputs_dict, self.Train_dict)

                    ### Optimizer and Log To Console: ###
                    self.Optimize_and_Log(total_loss, inputs_dict, self.outputs_dict)
                    ##########################################################

                    #############################################################################
                    ### TensorBoard: ###
                    self.Log_to_TensorBoard(inputs_dict)

                    ### Run Debug Callback: ###
                    with torch.no_grad():
                        self.Run_Debug_Calllback(inputs_dict)

                    ### Save Model Checkpoint: ###
                    self.Save_Network_Checkpoint(current_epoch, batch_index)
                    # self.Train_dict.current_time_step += 1  #for non-recursive batches current_time_step=0 by definition
                    #############################################################################

            toc('End of Epoch: ')

        ### Close TB Writter: ###
        if self.Train_dict.flag_write_TB:
            self.TB_object_train.TB_writer.export_scalars_to_json("./all_scalars.json")
            self.TB_object_train.TB_writer.close()
            self.TB_object_validation.TB_writer.export_scalars_to_json("./all_scalars.json")
            self.TB_object_validation.TB_writer.close()
        ################################################################################


    def _validate(self, Train_dict, device, test_dataloader, debug_callback=None):
        ### Model To Eval Mode: ###
        # torch.cuda.empty_cache()
        #TODO: again, there are problems with BatchNorm layer in eval mode...i hate batchnorm, try to understand what's happenning
        self.model.eval()
        # self.model.train()
        self.Train_dict.now_training = False
        self.Train_dict.current_time_step = 0

        self.Train_dict.current_time_step = 0
        ### Loop Over Val Data: ###
        if self.Train_dict.number_of_validation_samples == np.inf:
            #TODO: make this more accurate, probably at the dataset object level. for instance what if i'm using rolling_index but skipping frames
            self.Train_dict.number_of_validation_samples == test_dataloader.dataset.__len__()
        for batch_index_validation, data_from_dataloader_validation in enumerate(test_dataloader):
            if batch_index_validation < self.Train_dict.number_of_validation_samples:
                ### Get Data From DataLoader: ###
                print('validation: BATCH {}/{}:   '.format(batch_index_validation, min(self.Train_dict.number_of_validation_samples, self.Train_dict.num_mini_batches_val)))
                if self.Train_dict.flag_all_inputs_to_gpu:
                    data_from_dataloader_validation = data_from_dataloader_to_GPU(data_from_dataloader_validation, device)

                inputs_dict = EasyDict(data_from_dataloader_validation)
                inputs_dict.no_GT = self.Train_dict.no_GT
                with torch.no_grad():
                    total_loss = 0
                    ### Forward Input Through Model: ###
                    inputs_dict, model_output, self.outputs_dict = self.forward_propagation(self.Train_dict, inputs_dict)
                    ### Forward Model Results Through Loss: ###
                    self.total_loss_current, self.outputs_dict, inputs_dict = self.Loss_object.forward(inputs_dict, self.outputs_dict, self.Train_dict)
                    ### Accumulate Training Loss: ###
                    total_loss += self.total_loss_current
                    ### Get Final Total Loss: ###
                    self.outputs_dict.total_loss = total_loss

                    ### Validation TensorBoard IF VALIDATION DURING TRAINING!!!: ###
                    flag_training_and_i_want_to_write_TB = self.Train_dict.flag_write_TB and self.Train_dict.is_training
                    flag_inference_and_i_want_validation_TB_statistics = self.Train_dict.flag_do_validation and self.Train_dict.is_training == False
                    if (flag_training_and_i_want_to_write_TB) or (flag_inference_and_i_want_validation_TB_statistics):
                        self.TB_object_validation.get_TensorBoard_Metrics_Val(inputs_dict, self.outputs_dict)
                        # self.TB_object_validation.Log_TensorBoard_Val(self.Train_dict, model=self.model)  #TODO: i only want to log the averages...so probably should delete this

                    ### Debug Callback: ###
                    if debug_callback is not None:
                        debug_callback.run(batch_index_validation, inputs_dict, self.outputs_dict, self.Train_dict)

                    # self.Train_dict.current_time_step += 1
        ### Log Validation Averages To TensorBoard: ###
        if self.Train_dict.number_of_validation_samples != 0 and flag_training_and_i_want_to_write_TB:
            self.TB_object_validation.Log_TensorBoard_Val_Averages(self.Train_dict, model=self.model)

        ### TODO: add a section to get global statistics over validation: ###
        if hasattr(debug_callback, 'get_statistics'):
            debug_callback.get_statistics(inputs_dict, self.outputs_dict, self.Train_dict, batch_index_validation)

        ### Empty Cache: ###
        # torch.cuda.empty_cache()

    def inference(self):
        self.Train_dict.global_step = 0
        self._validate(self.Train_dict, self.device, self.test_dataloader, debug_callback=self.validation_callback)
############################################################################################################################################################################################



############################################################################################################################################################################################
# from RDND_proper.models.SwinIR.SwinIR import SwinIR
class AlternateTrainer(object):
    def __init__(self,
                 model,
                 preprocessing_object,
                 postprocessing_object,
                 Network_Optimizer_object,
                 Loss_object,
                 TB_object,
                 LR_scheduler_object,
                 Clip_gradient_object,
                 train_dataloader,
                 test_dataloader,
                 debug_callback,
                 validation_callback,
                 inference_callback,
                 Train_dict):

        self.initialize_internal_variables(model,
                                          preprocessing_object,
                                          postprocessing_object,
                                          Network_Optimizer_object,
                                          Loss_object,
                                          TB_object,
                                          LR_scheduler_object,
                                          Clip_gradient_object,
                                          train_dataloader,
                                          test_dataloader,
                                          debug_callback,
                                          validation_callback,
                                          inference_callback,
                                          Train_dict)

    def initialize_internal_variables(self,
                                      model,
                                      preprocessing_object,
                                      postprocessing_object,
                                      Network_Optimizer_object,
                                      Loss_object,
                                      TB_object,
                                      LR_scheduler_object,
                                      Clip_gradient_object,
                                      train_dataloader,
                                      test_dataloader,
                                      debug_callback,
                                      validation_callback,
                                      inference_callback,
                                      Train_dict):
        ### Get Wanted Network: ###
        self.Train_dict = Train_dict
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = Train_dict.device

        self.debug_callback = debug_callback
        self.validation_callback = validation_callback
        self.inference_callback = inference_callback

        self.model = model
        self.preprocessing_object = preprocessing_object
        self.postprocessing_object = postprocessing_object
        self.Loss_object = Loss_object
        self.Network_Optimizer_object = Network_Optimizer_object
        self.TB_object_train = TB_object[0]
        self.TB_object_validation = TB_object[1]
        self.LR_scheduler_object = LR_scheduler_object
        self.Clip_gradient_object = Clip_gradient_object
        self.debug_step = self.Train_dict.debug_step
        self.Network_checkpoint_folder = Train_dict.Network_checkpoint_folder
        self.outputs_dict = EasyDict()

    def initialize_Train_function(self, model):
        ####################################    Initialize Stuff For Training Loop:    ####################################
        ### Initialize Loop Steps: ###
        self.flag_stop_iterating = False
        self.Train_dict.global_batch_step = 0
        self.Train_dict.global_step = 0
        self.Train_dict.current_time_step = 0
        self.Train_dict.flag_first_save = True
        self.Train_dict.minimum_learning_rate_counter = 0
        self.Train_dict.Network_lr = self.Train_dict.init_learning_rate

        ### Network To Train Mode: ###
        model = model.train()
        return model

    def Optimize_and_Log(self, model, total_loss, inputs_dict, outputs_dict, network_optimizer):
        # ### Update Learning Rate: ###
        # self.Network_Optimizer, self.Train_dict = self.LR_scheduler_object.step(
        #     self.Network_Optimizer_object.Network_Optimizer, total_loss, self.Train_dict)

        ### Back-Propagate: ###
        total_loss.backward()

        # ### Clip Gradient Norm: ###
        # model, self.Train_dict = self.Clip_gradient_object.clip_gradient(model, self.Train_dict)

        ### Update Weights: ###
        network_optimizer.step()

        ### Print Network Description String: ###
        self.print_string(inputs_dict, outputs_dict, self.Train_dict)

        # ### Empty CUDA Cache Just In Case: ###
        # torch.cuda.empty_cache()

        return model, network_optimizer

    def Save_Network_Checkpoint(self, model, current_epoch, batch_index):
        if self.Train_dict.global_step % self.Train_dict.save_models_frequency == 0 or \
                (current_epoch == self.Train_dict.number_of_epochs - 1 and
                 batch_index + self.Train_dict.batch_size >= len(self.train_dataloader.train_dataset)):
            ### Save Network and Discriminator: ###
            print('Save Network and Discriminator Models:')
            print(model)
            print(self.Train_dict.Network_checkpoint_prefix)

            ### Decide Which Hyper-Parmaters To Save - From Project Or From Previous Checkpoint: ###
            current_Network_iteration_string, basic_Network_filename = \
                training_utils.Get_Network_Save_Basic_Names(self.Train_dict.global_step, self.Train_dict)
            self.Train_dict = \
                training_utils.Get_HyperParameters_Save_Flags(self.Train_dict.flag_first_save,
                                                              self.Train_dict)

            ### Save Dict: ###
            self.Train_dict.basic_Network_filename = basic_Network_filename
            self.Train_dict.variables_dictionary = None

            ### Save Network: ###
            training_utils.save_Network_parts_to_checkpoint(
                model, self.Network_checkpoint_folder,
                self.Network_Optimizer_object.Network_Optimizer,
                self.Train_dict)

            ### First Save Flag: ###
            self.Train_dict.flag_first_save = False
        return model

    def Log_to_TensorBoard(self, model, inputs_dict, outputs_dict):
        if self.Train_dict.flag_write_TB and (self.Train_dict.global_step % self.Train_dict.tensorboard_train_frequency == 0):
            self.TB_object_train.get_TensorBoard_Metrics_Train(inputs_dict, outputs_dict)
            self.TB_object_train.Log_TensorBoard_Train(self.Train_dict, model=model)
        return model, inputs_dict, outputs_dict

    def Run_Debug_Calllback(self, model, inputs_dict, outputs_dict):
        if self.debug_callback and (self.Train_dict.global_step % self.debug_step == 0):
            outputs_dict.model_state = model.state_dict()
            self.debug_callback.run(self.Train_dict.global_step, inputs_dict, outputs_dict, self.Train_dict)
        return model, inputs_dict, outputs_dict

    def inference(self, model):
        self.Train_dict.global_step = 0
        self._validate(model, self.Train_dict, self.device, self.test_dataloader, debug_callback=self.validation_callback)
        return model

    def Run_ValidationSet(self, model):
        if (self.Train_dict.global_step) % self.Train_dict.validation_set_frequency == 0 and self.Train_dict.flag_do_validation:
            print('########## Running validation set ##########')
            self._validate(model, self.Train_dict, self.device, self.test_dataloader, debug_callback=self.validation_callback)
            model.train()
        return model

    def forward_propagation(self, model, Train_dict, inputs_dict, outputs_dict):
        ### PreProcess the inputs: ###
        network_input, inputs_dict, self.Train_dict, model = self.preprocessing_object.forward(inputs_dict, self.Train_dict, model)

        ### Pass inputs to through the model: ###
        model_output = model.forward(network_input)
        outputs_dict.model_output = model_output

        ### Post process network's outputs: ###
        model_output_tensor, outputs_dict, inputs_dict, self.Train_dict, model = \
            self.postprocessing_object.forward(inputs_dict, outputs_dict, self.Train_dict, model)

        ### Return: ###
        return model, inputs_dict, model_output, outputs_dict


    def print_string(self, inputs_dict, outputs_dict, Train_dict):
        #TODO: change this to an object instead of an internal function
        ### General Stats: ###
        network_name_string = 'Network Name: ' + Train_dict.Network_checkpoint_prefix
        epoch_batch_string = 'EPOCH {:05d}, BATCH {}/{}:   '.format(Train_dict.current_epoch, Train_dict.batch_index, Train_dict.num_mini_batches_trn)

        ### Print String: ###
        print_string = ''
        print_string += network_name_string + ', ' + epoch_batch_string + ', '
        print_string += 'G Step ' + scientific_notation(Train_dict.global_step) + ', '
        # print_string += 'Gen lr: ' + scientific_notation(self.LR_scheduler_object.Network_lr) + ', \n'
        print_string += '\n'

        ### Losses (Project Specific): ###
        print_string += 'Total Loss: ' + scientific_notation(outputs_dict.total_loss.item()) + ', '

        ### Correct Training Indicators: ###
        #(*) i only search for (min,max) in first batch example to avoid too much calculation
        # network_final_output_min = outputs_dict.model_output['temporal_part_output_frame'][0].min().item()
        # network_final_output_max = outputs_dict.model_output['temporal_part_output_frame'][0].max().item()
        network_final_output_min = outputs_dict.network_output[0].min().item()
        network_final_output_max = outputs_dict.network_output[0].max().item()
        network_input_min = inputs_dict['center_frame_original'][0].min().item()
        network_input_max = inputs_dict['center_frame_original'][0].max().item()
        print_string += ' Network_Range=[' + decimal_notation(network_final_output_min) + ',' + decimal_notation(network_final_output_max) + '] , ' + \
                        ' GT_Range=[' + decimal_notation(network_input_min) + ',' + decimal_notation(network_input_max) + '] , '
        print_string += 'Grad-Norm: ' + scientific_notation(Train_dict.total_norm)
        toc(print_string)


    def Train(self):

        ### Initialize Train Function: ###
        model = self.initialize_Train_function(self.model)
        base_path = path_fix_path_for_linux('/media/mmm/DATADRIVE6/Omer')
        # model_path_LW = os.path.join(base_path, 'Pretrained Checkpoints/SwinIR/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth')
        # model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
        #                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
        #                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').cuda()
        # model.load_state_dict(torch.load(model_path_LW)['params'])

        outputs_dict = EasyDict()
        # network_optimizer = self.Network_Optimizer_object.Network_Optimizer
        network_optimizer = optim.Adam(model.parameters(), lr=0.001)

        ### Training Loop: ###
        for current_epoch in np.arange(0, self.Train_dict.number_of_epochs):
            # tic()
            ### If we got a signal to stop iterating then break: ###
            if self.flag_stop_iterating:
                break

            ### Empty CUDA Cache Just In Case: ###
            # torch.cuda.empty_cache()

            ### Loop over the current epoch data: ###
            tic()
            for batch_index, data_from_dataloader in enumerate(self.train_dataloader):
                toc('dataloading')
                tic()
                ### If we are at the end of our epoch or we got a signal to stop iterating -> break and stop: ###
                self.flag_stop_iterating = (self.Train_dict.global_batch_step == self.Train_dict.max_total_number_of_batches or self.flag_stop_iterating == True)
                if self.flag_stop_iterating:
                    break

                if self.flag_stop_iterating == False:
                    ##########################################################
                    # ### Empty CUDA Cache Just In Case: ###
                    # torch.cuda.empty_cache()

                    ### Training Flag: ###
                    self.Train_dict.now_training = True
                    ##########################################################

                    # ##########################################################
                    # ### Run Validation Set If Wanted: ###
                    # model = self.Run_ValidationSet(model)
                    # ##########################################################

                    ##########################################################
                    ### Uptick Steps: ###
                    self.Train_dict.global_step += 1
                    self.Train_dict.current_epoch = current_epoch
                    self.Train_dict.batch_index = batch_index
                    self.Train_dict.global_batch_step += 1  # total number of batches so far


                    ### Data From DataLoader To GPU & To EasyDict: ###
                    if self.Train_dict.flag_all_inputs_to_gpu:
                        data_from_dataloader = data_from_dataloader_to_GPU(data_from_dataloader, self.device)
                    inputs_dict = EasyDict(data_from_dataloader)

                    ### Pre-Process + Forward Pass + Post-Process: ###
                    # model, inputs_dict, model_output_tensor, outputs_dict = self.forward_propagation(model, self.Train_dict, inputs_dict, outputs_dict)

                    ### Get Train Loss: ###
                    # total_loss, outputs_dict, inputs_dict = self.Loss_object.forward(inputs_dict, outputs_dict, self.Train_dict)

                    ### Optimizer and Log To Console: ###
                    # model, network_optimizer = self.Optimize_and_Log(model, total_loss, inputs_dict, outputs_dict, network_optimizer)


                    #################################
                    ###### TODO - Delete, temp to understand what's wrong with the training: ########
                    ### PreProcess the inputs: ###
                    network_optimizer.zero_grad()
                    network_input, inputs_dict, self.Train_dict, model = self.preprocessing_object.forward(inputs_dict,
                                                                                                           self.Train_dict,
                                                                                                           model)


                    ### Pass inputs to through the model: ###
                    model_output = model.forward(network_input)
                    # outputs_dict.model_output = model_output

                    # ### Post process network's outputs: ###
                    # model_output_tensor, outputs_dict, inputs_dict, self.Train_dict, model = \
                    #     self.postprocessing_object.forward(inputs_dict, outputs_dict, self.Train_dict, model)

                    total_loss = (model_output - inputs_dict.center_frame_original).abs().mean()
                    total_loss.backward()
                    network_optimizer.step()
                    print(scientific_notation(total_loss.item()))
                    #################################



                    ##########################################################

                    #############################################################################
                    ### TensorBoard: ###
                    # model, inputs_dict, outputs_dict = self.Log_to_TensorBoard(model, inputs_dict, outputs_dict)

                    ### Run Debug Callback: ###
                    # model, inputs_dict, outputs_dict = self.Run_Debug_Calllback(model, inputs_dict, outputs_dict)

                    ### Save Model Checkpoint: ###
                    # model = self.Save_Network_Checkpoint(model, current_epoch, batch_index)
                    # self.Train_dict.current_time_step += 1  #for non-recursive batches current_time_step=0 by definition
                    #############################################################################

            toc('End of Epoch: ')

        ### Close TB Writter: ###
        if self.Train_dict.flag_write_TB:
            self.TB_object_train.TB_writer.export_scalars_to_json("./all_scalars.json")
            self.TB_object_train.TB_writer.close()
            self.TB_object_validation.TB_writer.export_scalars_to_json("./all_scalars.json")
            self.TB_object_validation.TB_writer.close()
        ################################################################################


    def _validate(self, model, Train_dict, device, test_dataloader, debug_callback=None):
        # ### Model To Eval Mode: ###
        # torch.cuda.empty_cache()
        #TODO: again, there are problems with BatchNorm layer in eval mode...i hate batchnorm, try to understand what's happenning
        model = model.eval()
        # self.model.train()
        self.Train_dict.now_training = False
        outputs_dict = EasyDict()

        ### Loop Over Val Data: ###
        for batch_index_validation, data_from_dataloader_validation in enumerate(test_dataloader):
            if batch_index_validation < self.Train_dict.number_of_validation_samples:
                ### Get Data From DataLoader: ###
                print('validation: BATCH {}/{}:   '.format(batch_index_validation, self.Train_dict.num_mini_batches_val))
                if self.Train_dict.flag_all_inputs_to_gpu:  #TODO: insert flag_all_inputs_to_gpu into data_from_dataloader_to_GPU function?
                    data_from_dataloader_validation = data_from_dataloader_to_GPU(data_from_dataloader_validation, device)
                inputs_dict = EasyDict(data_from_dataloader_validation)

                with torch.no_grad():
                    total_loss = 0
                    ### Forward Input Through Model: ###
                    model, inputs_dict, model_output, outputs_dict = self.forward_propagation(model, self.Train_dict, inputs_dict, outputs_dict)
                    ### Forward Model Results Through Loss: ###
                    #TODO: don't have loss as an internal variables/attribute
                    total_loss_current, outputs_dict, inputs_dict = self.Loss_object.forward(inputs_dict, outputs_dict, self.Train_dict)
                    ### Accumulate Training Loss: ###
                    total_loss += total_loss_current
                    ### Get Final Total Loss: ###
                    outputs_dict.total_loss = total_loss

                    ### Validation TensorBoard IF VALIDATION DURING TRAINING!!!: ###
                    flag_training_and_i_want_to_write_TB = self.Train_dict.flag_write_TB and self.Train_dict.is_training
                    flag_inference_and_i_want_validation_TB_statistics = self.Train_dict.flag_do_validation and self.Train_dict.is_training == False
                    if (flag_training_and_i_want_to_write_TB) or (flag_inference_and_i_want_validation_TB_statistics):
                        self.TB_object_validation.get_TensorBoard_Metrics_Val(inputs_dict, outputs_dict)
                        # self.TB_object_validation.Log_TensorBoard_Val(self.Train_dict, model=self.model)  #TODO: i only want to log the averages...so probably should delete this

                    ### Debug Callback: ###
                    if debug_callback is not None:
                        debug_callback.run(batch_index_validation, inputs_dict, outputs_dict, self.Train_dict)

        ### Log Validation Averages To TensorBoard: ###
        if self.Train_dict.number_of_validation_samples != 0 and flag_training_and_i_want_to_write_TB:
            self.TB_object_validation.Log_TensorBoard_Val_Averages(self.Train_dict, model=model)

        ### TODO: add a section to get global statistics over validation: ###
        if hasattr(debug_callback, 'get_statistics'):
            debug_callback.get_statistics(inputs_dict, outputs_dict, self.Train_dict, batch_index_validation)

        # ### Empty Cache: ###
        # torch.cuda.empty_cache

    def inference(self, model):
        self.Train_dict.global_step = 0
        model = self._validate(model, self.Train_dict, self.device, self.test_dataloader, debug_callback=self.validation_callback)
        return model
############################################################################################################################################################################################




############################################################################################################################################################################################
class GeneralTrainer_Recursive(GeneralTrainer):
    def __init__(self,
                 model,
                 preprocessing_object,
                 postprocessing_object,
                 Network_Optimizer_object,
                 Loss_object,
                 TB_object,
                 LR_scheduler_object,
                 Clip_gradient_object,
                 train_dataloader,
                 test_dataloader,
                 debug_callback,
                 validation_callback,
                 inference_callback,
                 Train_dict):

        self.initialize_internal_variables(model,
                 preprocessing_object,
                 postprocessing_object,
                 Network_Optimizer_object,
                 Loss_object,
                 TB_object,
                 LR_scheduler_object,
                 Clip_gradient_object,
                 train_dataloader,
                 test_dataloader,
                 debug_callback,
                 validation_callback,
                 inference_callback,
                 Train_dict)

    def Train(self):
        ### Initialize Train Function: ###
        self.initialize_Train_function()

        ### Training Loop: ###
        for current_epoch in np.arange(0, self.Train_dict.number_of_epochs):
            # tic()
            ### If we got a signal to stop iterating then break: ###
            if self.flag_stop_iterating:
                break

            ### Empty CUDA Cache Just In Case: ###
            # torch.cuda.empty_cache()

            ### Loop over the current epoch data: ###
            for batch_index, data_from_dataloader in enumerate(self.train_dataloader):

                ##########################################################
                ### Initialize Things Before Forward: ###
                self.flag_stop_iterating = (self.Train_dict.global_batch_step == self.Train_dict.max_total_number_of_batches or self.flag_stop_iterating == True)
                if self.flag_stop_iterating:
                    break
                # torch.cuda.empty_cache()
                self.Train_dict.now_training = True
                ##########################################################

                ##########################################################
                ### Run Validation Set: ###
                self.Run_ValidationSet()
                ##########################################################

                #############################################################################
                ### Uptick Steps: ###
                self.Train_dict.global_step += 1
                self.Train_dict.current_epoch = current_epoch
                self.Train_dict.batch_index = batch_index
                self.Train_dict.global_batch_step += 1  # total number of batches so far

                ### Initialize Recursion Counters: ###
                self.Train_dict.number_of_same_image_backward_steps = 0
                self.Train_dict.number_of_time_steps_from_batch_start = 0

                ### Get Proper Flags For Memory Cells For Initialization (Initialize with Input/Zero): ###
                self.Train_dict.reset_hidden_states_flag = 0

                ### Data From DataLoader To GPU & To EasyDict: ###
                if self.Train_dict.flag_all_inputs_to_gpu:
                    data_from_dataloader = data_from_dataloader_to_GPU(data_from_dataloader, self.device)
                inputs_dict = EasyDict(data_from_dataloader)
                ##########################################################

                ##########################################################
                ### Deleting Stuff To Prevent Memory Leak: ###
                self.outputs_dict = EasyDict()

                ### Loop Over Backward Steps: ###
                output_batch_size = data_from_dataloader['output_frames_original'].shape #TODO: this is the ONE PLACE where the trainer is not totally "general"...think about it
                while output_batch_size[0] == self.Train_dict.batch_size and self.flag_stop_iterating == False \
                        and self.Train_dict.number_of_same_image_backward_steps < self.Train_dict.number_of_total_backward_steps_per_image:

                    #############################################################################
                    ### Empty CUDA Cache Just In Case: ###
                    # torch.cuda.empty_cache()

                    ### DETACH by default after every number_of_time_steps_before_backward of graph accumulation: ###
                    if self.Train_dict.number_of_same_image_backward_steps > 0:
                        self.Train_dict.reset_hidden_states_flag = 2

                    ### Uptick Steps: ###
                    self.Train_dict.number_of_same_image_backward_steps += 1

                    ### Zero Network Grad: ###
                    self.Network_Optimizer_object.Network_Optimizer.zero_grad()
                    #############################################################################

                    ### Loop Over Time-Steps: ###
                    total_loss = 0
                    # torch.cuda.empty_cache()
                    for current_time_step in arange(0, self.Train_dict.number_of_time_steps_in_one_backward):
                        ### Pre-Process + Forward Pass + Post-Process: ###
                        self.Train_dict.current_time_step = current_time_step
                        inputs_dict, model_output_tensor, self.outputs_dict = self.forward_propagation(self.Train_dict, inputs_dict)

                        ### In any case after whatever initial initialization signal -> do nothing and continue accumulating graph: ###
                        self.Train_dict.reset_hidden_states_flag = 1

                        ### Get Train Loss: ###
                        total_loss_current, self.outputs_dict, inputs_dict = self.Loss_object.forward(inputs_dict, self.outputs_dict, self.Train_dict)

                        ### Accumulate Training Loss: ###
                        total_loss += total_loss_current

                        ### Uptick Counters: ###
                        self.Train_dict.number_of_time_steps_from_batch_start += 1

                    ### Empty CUDA Cache Just In Case: ###
                    # torch.cuda.empty_cache()

                    ### Optimizer and Log To Console: ###
                    self.Optimize_and_Log(total_loss, inputs_dict, self.outputs_dict)

                    ### END OF CURRENT BACKWARD STEP ITERATIONS ###
                ### END OF CURRENT BATCH BACKWARD STEPS!

                #############################################################################
                ### TensorBoard: ###
                self.Log_to_TensorBoard(inputs_dict)

                ### Run Debug Callback: ###
                self.Run_Debug_Calllback(inputs_dict)

                ### Save Model Checkpoint: ###
                self.Save_Network_Checkpoint(current_epoch, batch_index)
                #############################################################################

            toc('End of Epoch: ')

        ### Close TB Writter: ###
        if self.Train_dict.flag_write_TB:
            self.TB_object_validation.TB_writer.export_scalars_to_json("./all_scalars.json")
            self.TB_object_validation.TB_writer.close()
            self.TB_object_train.TB_writer.export_scalars_to_json("./all_scalars.json")
            self.TB_object_train.TB_writer.close()

    def _validate(self, Train_dict, device, test_dataloader, debug_callback=None):
        ### Model To Eval Mode: ###
        # #TODO: be careful when using batchnorm because then you need to also freeze them and switch to gathered stats etc'
        with torch.no_grad():

            # torch.cuda.empty_cache()
            self.model.eval()
            self.Train_dict.now_training = False
            self.Train_dict.current_time_step = 0

            ### Get Proper Flags For Memory Cells In Case Of Movie (every batch is actually just a different frame): ###
            if self.Train_dict.reset_every_batch == False:
                self.Train_dict.reset_hidden_states_flag = 1
            else:
                self.Train_dict.reset_hidden_states_flag = 0

            ### Loop Over Val Data: ###
            for batch_index_validation, data_from_dataloader_validation in enumerate(test_dataloader):
                if batch_index_validation < self.Train_dict.number_of_validation_samples:
                    ### Initialize Recursion Counters: ###
                    self.Train_dict.number_of_same_image_backward_steps = 0
                    self.Train_dict.number_of_time_steps_from_batch_start = 0

                    ### Get Data From DataLoader: ###
                    print('validation: BATCH {}/{}:   '.format(batch_index_validation, self.Train_dict.num_mini_batches_val))
                    if self.Train_dict.flag_all_inputs_to_gpu:
                        data_from_dataloader_validation = data_from_dataloader_to_GPU(data_from_dataloader_validation, device)
                    inputs_dict = EasyDict(data_from_dataloader_validation)

                    with torch.no_grad():
                        ### Get Proper Flags For Memory Cells (Signaling Start Of Batch): ###
                        if self.Train_dict.reset_every_batch:
                            self.Train_dict.reset_hidden_states_flag = 0

                        ### Loop Over Time Dim Of Current Batch: ###
                        total_loss = 0
                        total_number_of_forward_steps = Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward
                        for current_time_step in np.arange(total_number_of_forward_steps):
                            #print(current_time_step)
                            ### Forward Data Into Model: ###
                            self.Train_dict.current_time_step = current_time_step
                            inputs_dict, model_output, self.outputs_dict = self.forward_propagation(self.Train_dict, inputs_dict)
                            self.total_loss_current, self.outputs_dict, inputs_dict = self.Loss_object.forward(inputs_dict, self.outputs_dict, self.Train_dict)
                            ### Accumulate Training Loss: ###
                            total_loss += self.total_loss_current
                            ### Uptick Counters: ###
                            self.Train_dict.number_of_time_steps_from_batch_start += 1
                        ### Get Final Total Loss: ###
                        self.outputs_dict.total_loss = total_loss

                        ### Validation TensorBoard IF VALIDATION DURING TRAINING!!!: ###
                        flag_training_and_i_want_to_write_TB = self.Train_dict.flag_write_TB and self.Train_dict.is_training
                        flag_inference_and_i_want_validation_TB_statistics = self.Train_dict.flag_do_validation and self.Train_dict.is_training == False
                        if (flag_training_and_i_want_to_write_TB) or (flag_inference_and_i_want_validation_TB_statistics):
                            self.TB_object_validation.get_TensorBoard_Metrics_Val(inputs_dict, self.outputs_dict)
                            # self.TB_object_validation.Log_TensorBoard_Val(Train_dict, model=self.model)  #TODO: i am commenting

                        ### Debug Callback: ### #TODO: should we have a Val_Callback in addition to Debug_Callback?
                        if debug_callback is not None:
                            debug_callback.run(batch_index_validation, inputs_dict, self.outputs_dict, self.Train_dict)

                        self.Train_dict.current_time_step += 1
            ### Log Validation Averages To TensorBoard: ###
            if self.Train_dict.number_of_validation_samples != 0 and flag_training_and_i_want_to_write_TB:
                self.TB_object_validation.Log_TensorBoard_Val_Averages(self.Train_dict, model=self.model)

            ### TODO: add a section to get global statistics over validation: ###
            if hasattr(debug_callback, 'get_statistics'):
                debug_callback.get_statistics(inputs_dict, self.outputs_dict, self.Train_dict, batch_index_validation)

            ### Empty Cache: ###
            # torch.cuda.empty_cache()

############################################################################################################################################################################################


