from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from RDND_proper.models.star_flow.models.pwc_modules import conv, upsample2d_as, rescale_flow, initialize_msra
from RDND_proper.models.star_flow.models.pwc_modules import WarpingLayer, FeatureExtractor
from RDND_proper.models.star_flow.models.pwc_modules import FlowAndOccContextNetwork, FlowAndOccEstimatorDense
from RDND_proper.models.star_flow.models.irr_modules import OccUpsampleNetwork, RefineFlow, RefineOcc
from RDND_proper.models.star_flow.models.correlation_package.correlation import CorrTorch as Correlation

import copy



class StarFlow(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(StarFlow, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.training = True

        ### Feature Extractor For Each Image: ###
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        ### Warping Layer: ###  #TODO: test against my implementation
        self.warping_layer = WarpingLayer()

        ### Get number of correlation tensor channels (2*search_range+1) for each axis: ###
        self.dim_corr = (self.search_range * 2 + 1) ** 2

        ### Get number of channels for the subsequent networks: ###
        self.num_ch_in = self.dim_corr + 32 + 2 + 1  #TODO: understand why +32+2+1????

        ### Get Different Auxiliary Networks: ###
        self.flow_and_occ_estimators = FlowAndOccEstimatorDense(2 * self.num_ch_in)
        self.context_networks = FlowAndOccContextNetwork(2 * self.num_ch_in + 448 + 2 + 1)  #TODO: understand the 448+2+1
        self.occ_shuffle_upsample = OccUpsampleNetwork(11, 1)  #TODO: understand the 11,1. 1 occlusion channel, where does the 10 (11-1) comes from?

        ### Get the different 1x1 convs: ###
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1)])
        self.conv_1x1_1 = conv(16, 3, kernel_size=1, stride=1, dilation=1)
        self.conv_1x1_time = conv(2 * self.num_ch_in + 448, self.num_ch_in, kernel_size=1, stride=1, dilation=1)

        ### Get Flow & Occlusion Refinement Networks: ###
        self.refine_flow = RefineFlow(2 + 1 + 32)
        self.refine_occ = RefineOcc(1 + 32 + 32)

        initialize_msra(self.modules())

    def forward(self, network_input):
        list_imgs = network_input
        _, _, height_im, width_im = list_imgs[0].size()

        ### Get Feature Pyramids For Each Image: ###
        #(*). self.feature_pyramid_extractor returns a list of features and i concatenate the original image
        list_pyramids = []  # indices : [time][level]
        for im in list_imgs:
            list_pyramids.append(self.feature_pyramid_extractor(im) + [im])    #TODO: turn to simply two images
        number_of_pyramid_scales = len(list_pyramids[0])

        ### Initialize Outputs: ###
        output_dict = {}
        output_dict_eval = {}
        flows_f = []  # indices : [level][time]
        flows_b = []  # indices : [level][time]
        occs_f = []
        occs_b = []
        flows_coarse_f = []
        occs_coarse_f = []
        for pyramid_index in range(number_of_pyramid_scales):
            flows_f.append([])
            flows_b.append([])
            occs_f.append([])
            occs_b.append([])
        for l in range(self.output_level + 1):  #TODO: understand what this means!!!!
            flows_coarse_f.append([])
            occs_coarse_f.append([])

        ### Initialize Varaiables Holding Current Flow Estimation: ###
        b_size, _, h_x1, w_x1, = list_pyramids[0][0].size()
        init_dtype = list_pyramids[0][0].dtype
        init_device = list_pyramids[0][0].device
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        occ_f = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        previous_features = []


        for image_index in range(len(list_imgs) - 1):
            ### Get Current Image Pyramids: ###
            x1_pyramid, x2_pyramid = list_pyramids[image_index:image_index + 2]   #TODO: switch to only 2 images

            for pyramid_level_index, (x1_current_features, x2_current_features) in enumerate(zip(x1_pyramid, x2_pyramid)):

                if pyramid_level_index <= self.output_level:
                    ### For the first image initialize previous_features to zeros: ###
                    if image_index == 0:
                        bs_, _, h_, w_, = list_pyramids[0][pyramid_level_index].size()
                        previous_features.append(torch.zeros(bs_, self.num_ch_in, h_, w_, dtype=init_dtype, device=init_device).float())

                    ### Warp current features according to previous flow: ###
                    if pyramid_level_index == 0:
                        x2_current_features_warp = x2_current_features
                        x1_current_features_warp = x1_current_features
                    else:
                        flow_f = upsample2d_as(flow_f, x1_current_features, mode="bilinear")
                        flow_b = upsample2d_as(flow_b, x2_current_features, mode="bilinear")
                        occ_f = upsample2d_as(occ_f, x1_current_features, mode="bilinear")
                        occ_b = upsample2d_as(occ_b, x2_current_features, mode="bilinear")
                        x2_current_features_warp = self.warping_layer(x2_current_features, flow_f, height_im, width_im, self._div_flow)
                        x1_current_features_warp = self.warping_layer(x1_current_features, flow_b, height_im, width_im, self._div_flow)

                    ### Get Correlation Tensors For Current Pyramid Layer Features: ###
                    out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1,
                                             max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1_current_features, x2_current_features_warp)
                    out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1,
                                             max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2_current_features, x1_current_features_warp)
                    out_corr_relu_f = self.leakyRELU(out_corr_f)
                    out_corr_relu_b = self.leakyRELU(out_corr_b)

                    ### Pass Current Features Through 1x1 Conv To Change Number Of Channels: ###
                    if pyramid_level_index != self.output_level:
                        x1_current_features_1by1 = self.conv_1x1[pyramid_level_index](x1_current_features)
                        x2_current_features_1by1 = self.conv_1x1[pyramid_level_index](x2_current_features)
                    else:
                        x1_current_features_1by1 = x1_current_features
                        x2_current_features_1by1 = x2_current_features

                    ### Warp Previous Features (Of Previous Image) Of Current Scale According To Flow Estimated: ###  #TODO: understand when was this flow estimated
                    if image_index > 0:  # temporal connection:
                        previous_features[pyramid_level_index] = self.warping_layer(previous_features[pyramid_level_index],
                                                                  flows_b[pyramid_level_index][-1], height_im, width_im, self._div_flow)

                    ### Rescale Forward & Backward Flows (according to new scale???): ###
                    flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
                    flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

                    ### Concat Different Tensors Before Passing Them Into The Current Flow+Occ Network Estimator: ###
                    features = torch.cat([previous_features[pyramid_level_index], out_corr_relu_f, x1_current_features_1by1, flow_f, occ_f], 1)
                    features_b = torch.cat([torch.zeros_like(previous_features[pyramid_level_index]), out_corr_relu_b, x2_current_features_1by1, flow_b, occ_b], 1)

                    ### Get Flow Residual, Occlusion Residual and Warped Features From Network (for both forward and backward): ###
                    x_intm_f, flow_res_f, occ_res_f = self.flow_and_occ_estimators(features)
                    flow_est_f = flow_f + flow_res_f
                    occ_est_f = occ_f + occ_res_f
                    with torch.no_grad():
                        x_intm_b, flow_res_b, occ_res_b = self.flow_and_occ_estimators(features_b)
                        flow_est_b = flow_b + flow_res_b
                        occ_est_b = occ_b + occ_res_b

                    ### Pass Features, Flow and Occlusions Through A Context Network For Further Refinement Of Flow & Occlusions: ###
                    flow_cont_res_f, occ_cont_res_f = self.context_networks(torch.cat([x_intm_f, flow_est_f, occ_est_f], dim=1))
                    flow_cont_f = flow_est_f + flow_cont_res_f
                    occ_cont_f = occ_est_f + occ_cont_res_f
                    with torch.no_grad():
                        flow_cont_res_b, occ_cont_res_b = self.context_networks(torch.cat([x_intm_b, flow_est_b, occ_est_b], dim=1))
                        flow_cont_b = flow_est_b + flow_cont_res_b
                        occ_cont_b = occ_est_b + occ_cont_res_b

                    ### Upscale (Practically DownSize I Think...) Current Images and Warp Them According To Current Flow Estimations: ###
                    img1_resize = upsample2d_as(list_imgs[image_index], flow_f, mode="bilinear")
                    img2_resize = upsample2d_as(list_imgs[image_index + 1], flow_b, mode="bilinear")
                    img2_warp = self.warping_layer(img2_resize,
                                                   rescale_flow(flow_cont_f, self._div_flow, width_im, height_im,
                                                                to_local=False), height_im, width_im, self._div_flow)
                    img1_warp = self.warping_layer(img1_resize,
                                                   rescale_flow(flow_cont_b, self._div_flow, width_im, height_im,
                                                                to_local=False), height_im, width_im, self._div_flow)

                    ### Refine Flows According To Image Features & Image Differences: ###
                    flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_current_features_1by1)
                    flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_current_features_1by1)

                    ### Rescale Flows: ###
                    flow_cont_f = rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False)
                    flow_cont_b = rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False)
                    flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)
                    flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

                    ### Occlusion Refinement: ###
                    x2_current_features_1by1_warp = self.warping_layer(x2_current_features_1by1, flow_f, height_im, width_im, self._div_flow)
                    x1_current_features_1by1_warp = self.warping_layer(x1_current_features_1by1, flow_b, height_im, width_im, self._div_flow)
                    occ_f = self.refine_occ(occ_cont_f.detach(), x1_current_features_1by1, x1_current_features_1by1 - x2_current_features_1by1_warp)
                    occ_b = self.refine_occ(occ_cont_b.detach(), x2_current_features_1by1, x2_current_features_1by1 - x1_current_features_1by1_warp)

                    ### Save Features For Temporal Connection: ###
                    previous_features[pyramid_level_index] = self.conv_1x1_time(x_intm_f)
                    flows_f[pyramid_level_index].append(flow_f)
                    occs_f[pyramid_level_index].append(occ_f)
                    flows_b[pyramid_level_index].append(flow_b)
                    occs_b[pyramid_level_index].append(occ_b)
                    flows_coarse_f[pyramid_level_index].append(flow_cont_f)
                    occs_coarse_f[pyramid_level_index].append(occ_cont_f)
                    # flows.append([flow_cont_f, flow_cont_b, flow_f, flow_b])
                    # occs.append([occ_cont_f, occ_cont_b, occ_f, occ_b])

                else:   # if pyramid_level_index > self.output_level:
                    flow_f = upsample2d_as(flow_f, x1_current_features, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2_current_features, mode="bilinear")
                    flows_f[pyramid_level_index].append(flow_f)
                    flows_b[pyramid_level_index].append(flow_b)
                    # flows.append([flow_f, flow_b])

                    ### Warp Current Feautres and Flow Estimations: ###
                    x2_current_features_warp = self.warping_layer(x2_current_features, flow_f, height_im, width_im, self._div_flow)
                    x1_current_features_warp = self.warping_layer(x1_current_features, flow_b, height_im, width_im, self._div_flow)
                    flow_b_warp = self.warping_layer(flow_b, flow_f, height_im, width_im, self._div_flow)
                    flow_f_warp = self.warping_layer(flow_f, flow_b, height_im, width_im, self._div_flow)

                    #### Pass Features & Warped Features Through 1x1 Conv: ###
                    if pyramid_level_index != self.num_levels - 1:
                        x1_current_features_in = self.conv_1x1_1(x1_current_features)
                        x2_current_features_in = self.conv_1x1_1(x2_current_features)
                        x1_current_features_w_in = self.conv_1x1_1(x1_current_features_warp)
                        x2_current_features_w_in = self.conv_1x1_1(x2_current_features_warp)
                    else:
                        x1_current_features_in = x1_current_features
                        x2_current_features_in = x2_current_features
                        x1_current_features_w_in = x1_current_features_warp
                        x2_current_features_w_in = x2_current_features_warp

                    ### Pass Occlusions, Features, Flows and Warped Flows Estimations Through Network For Better Occlusion Estimation: ###
                    occ_f = self.occ_shuffle_upsample(occ_f, torch.cat([x1_current_features_in, x2_current_features_w_in, flow_f, flow_b_warp], dim=1))
                    occ_b = self.occ_shuffle_upsample(occ_b, torch.cat([x2_current_features_in, x1_current_features_w_in, flow_b, flow_f_warp], dim=1))

                    ### Update Occlusions Pyramid: ###
                    occs_f[pyramid_level_index].append(occ_f)
                    occs_b[pyramid_level_index].append(occ_b)
                    # occs.append([occ_f, occ_b])

            ### After Looping Over All Pyramid Levels Assign Zeros To Current Flows and Occlusions Estimations: ###
            flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            occ_f = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
            occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()


        if self.training:
            if len(list_imgs) > 2:
                for l in range(len(flows_f)):
                    flows_f[l] = torch.stack(flows_f[l], 0)
                    occs_f[l] = torch.stack(occs_f[l], 0)
                for l in range(len(flows_coarse_f)):
                    flows_coarse_f[l] = torch.stack(flows_coarse_f[l], 0)
                    occs_coarse_f[l] = torch.stack(occs_coarse_f[l], 0)
            else:
                for l in range(len(flows_f)):
                    flows_f[l] = flows_f[l][0]
                    occs_f[l] = occs_f[l][0]
                for l in range(len(flows_coarse_f)):
                    flows_coarse_f[l] = flows_coarse_f[l][0]
                    occs_coarse_f[l] = occs_coarse_f[l][0]
            output_dict['flow'] = flows_f
            output_dict['occ'] = occs_f
            output_dict['flow_coarse'] = flows_coarse_f
            output_dict['occ_coarse'] = occs_coarse_f
            return output_dict
        else:
            output_dict_eval = {}
            if len(list_imgs) > 2:
                out_flow = []
                out_occ = []
                for i in range(len(flows_f[0])):
                    out_flow.append(
                        upsample2d_as(flows_f[-1][i], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow))
                    out_occ.append(upsample2d_as(occs_f[-1][i], list_imgs[0], mode="bilinear"))
                out_flow = torch.stack(out_flow, 0)
                out_occ = torch.stack(out_occ, 0)
            else:
                out_flow = upsample2d_as(flows_f[-1][0], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow)
                out_occ = upsample2d_as(occs_f[-1][0], list_imgs[0], mode="bilinear")
            output_dict_eval['flow'] = out_flow
            output_dict_eval['occ'] = out_occ
            return output_dict_eval


    # def forward_multiple(self, network_input):
    #
    #     # if 'input_images' in input_dict.keys():
    #     #     list_imgs = input_dict['input_images']
    #     # else:
    #     #     x1_raw = input_dict['input1']
    #     #     x2_raw = input_dict['input2']
    #     #     list_imgs = [x1_raw, x2_raw]
    #
    #     list_imgs = network_input
    #     _, _, height_im, width_im = list_imgs[0].size()
    #
    #     # on the bottom level are original images
    #     list_pyramids = [] #indices : [time][level]
    #     for im in list_imgs:
    #         list_pyramids.append(self.feature_pyramid_extractor(im) + [im])
    #
    #     # outputs
    #     output_dict = {}
    #     output_dict_eval = {}
    #     flows_f = [] #indices : [level][time]
    #     flows_b = [] #indices : [level][time]
    #     occs_f = []
    #     occs_b = []
    #     flows_coarse_f = []
    #     occs_coarse_f = []
    #     for l in range(len(list_pyramids[0])):
    #         flows_f.append([])
    #         flows_b.append([])
    #         occs_f.append([])
    #         occs_b.append([])
    #     for l in range(self.output_level + 1):
    #         flows_coarse_f.append([])
    #         occs_coarse_f.append([])
    #
    #     # init
    #     b_size, _, h_x1, w_x1, = list_pyramids[0][0].size()
    #     init_dtype = list_pyramids[0][0].dtype
    #     init_device = list_pyramids[0][0].device
    #     flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #     flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #     occ_f = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #     occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #     previous_features = []
    #
    #     for i in range(len(list_imgs) - 1):
    #         x1_pyramid, x2_pyramid = list_pyramids[i:i+2]
    #
    #         for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
    #
    #             if l <= self.output_level:
    #                 if i == 0:
    #                     bs_, _, h_, w_, = list_pyramids[0][l].size()
    #                     previous_features.append(torch.zeros(bs_, self.num_ch_in, h_, w_, dtype=init_dtype, device=init_device).float())
    #
    #                 # warping
    #                 if l == 0:
    #                     x2_warp = x2
    #                     x1_warp = x1
    #                 else:
    #                     flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
    #                     flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
    #                     occ_f = upsample2d_as(occ_f, x1, mode="bilinear")
    #                     occ_b = upsample2d_as(occ_b, x2, mode="bilinear")
    #                     x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
    #                     x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
    #
    #                 # correlation
    #                 out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
    #                 out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
    #                 out_corr_relu_f = self.leakyRELU(out_corr_f)
    #                 out_corr_relu_b = self.leakyRELU(out_corr_b)
    #
    #                 if l != self.output_level:
    #                     x1_1by1 = self.conv_1x1[l](x1)
    #                     x2_1by1 = self.conv_1x1[l](x2)
    #                 else:
    #                     x1_1by1 = x1
    #                     x2_1by1 = x2
    #
    #                 if i > 0: #temporal connection:
    #                     previous_features[l] = self.warping_layer(previous_features[l],
    #                                     flows_b[l][-1], height_im, width_im, self._div_flow)
    #
    #                 # Flow and occlusions estimation
    #                 flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
    #                 flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)
    #
    #                 features = torch.cat([previous_features[l], out_corr_relu_f, x1_1by1, flow_f, occ_f], 1)
    #                 features_b = torch.cat([torch.zeros_like(previous_features[l]), out_corr_relu_b, x2_1by1, flow_b, occ_b], 1)
    #
    #                 x_intm_f, flow_res_f, occ_res_f = self.flow_and_occ_estimators(features)
    #                 flow_est_f = flow_f + flow_res_f
    #                 occ_est_f = occ_f + occ_res_f
    #                 with torch.no_grad():
    #                     x_intm_b, flow_res_b, occ_res_b = self.flow_and_occ_estimators(features_b)
    #                     flow_est_b = flow_b + flow_res_b
    #                     occ_est_b = occ_b + occ_res_b
    #
    #                 # Context:
    #                 flow_cont_res_f, occ_cont_res_f = self.context_networks(torch.cat([x_intm_f, flow_est_f, occ_est_f], dim=1))
    #                 flow_cont_f = flow_est_f + flow_cont_res_f
    #                 occ_cont_f = occ_est_f + occ_cont_res_f
    #                 with torch.no_grad():
    #                     flow_cont_res_b, occ_cont_res_b = self.context_networks(torch.cat([x_intm_b, flow_est_b, occ_est_b], dim=1))
    #                     flow_cont_b = flow_est_b + flow_cont_res_b
    #                     occ_cont_b = occ_est_b + occ_cont_res_b
    #
    #                 # refinement
    #                 img1_resize = upsample2d_as(list_imgs[i], flow_f, mode="bilinear")
    #                 img2_resize = upsample2d_as(list_imgs[i+1], flow_b, mode="bilinear")
    #                 img2_warp = self.warping_layer(img2_resize, rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False), height_im, width_im, self._div_flow)
    #                 img1_warp = self.warping_layer(img1_resize, rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False), height_im, width_im, self._div_flow)
    #
    #                 # flow refine
    #                 flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_1by1)
    #                 flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_1by1)
    #
    #                 flow_cont_f = rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False)
    #                 flow_cont_b = rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False)
    #                 flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)
    #                 flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)
    #
    #                 # occ refine
    #                 x2_1by1_warp = self.warping_layer(x2_1by1, flow_f, height_im, width_im, self._div_flow)
    #                 x1_1by1_warp = self.warping_layer(x1_1by1, flow_b, height_im, width_im, self._div_flow)
    #
    #                 occ_f = self.refine_occ(occ_cont_f.detach(), x1_1by1, x1_1by1 - x2_1by1_warp)
    #                 occ_b = self.refine_occ(occ_cont_b.detach(), x2_1by1, x2_1by1 - x1_1by1_warp)
    #
    #                 # save features for temporal connection:
    #                 previous_features[l] = self.conv_1x1_time(x_intm_f)
    #                 flows_f[l].append(flow_f)
    #                 occs_f[l].append(occ_f)
    #                 flows_b[l].append(flow_b)
    #                 occs_b[l].append(occ_b)
    #                 flows_coarse_f[l].append(flow_cont_f)
    #                 occs_coarse_f[l].append(occ_cont_f)
    #                 #flows.append([flow_cont_f, flow_cont_b, flow_f, flow_b])
    #                 #occs.append([occ_cont_f, occ_cont_b, occ_f, occ_b])
    #
    #             else:
    #                 flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
    #                 flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
    #                 flows_f[l].append(flow_f)
    #                 flows_b[l].append(flow_b)
    #                 #flows.append([flow_f, flow_b])
    #
    #                 x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
    #                 x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
    #                 flow_b_warp = self.warping_layer(flow_b, flow_f, height_im, width_im, self._div_flow)
    #                 flow_f_warp = self.warping_layer(flow_f, flow_b, height_im, width_im, self._div_flow)
    #
    #                 if l != self.num_levels-1:
    #                     x1_in = self.conv_1x1_1(x1)
    #                     x2_in = self.conv_1x1_1(x2)
    #                     x1_w_in = self.conv_1x1_1(x1_warp)
    #                     x2_w_in = self.conv_1x1_1(x2_warp)
    #                 else:
    #                     x1_in = x1
    #                     x2_in = x2
    #                     x1_w_in = x1_warp
    #                     x2_w_in = x2_warp
    #
    #                 occ_f = self.occ_shuffle_upsample(occ_f, torch.cat([x1_in, x2_w_in, flow_f, flow_b_warp], dim=1))
    #                 occ_b = self.occ_shuffle_upsample(occ_b, torch.cat([x2_in, x1_w_in, flow_b, flow_f_warp], dim=1))
    #
    #                 occs_f[l].append(occ_f)
    #                 occs_b[l].append(occ_b)
    #                 #occs.append([occ_f, occ_b])
    #
    #         flow_f = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #         flow_b = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #         occ_f = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #         occ_b = torch.zeros(b_size, 1, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
    #
    #     if self.training:
    #         if len(list_imgs) > 2:
    #             for l in range(len(flows_f)):
    #                 flows_f[l] = torch.stack(flows_f[l], 0)
    #                 occs_f[l] = torch.stack(occs_f[l], 0)
    #             for l in range(len(flows_coarse_f)):
    #                 flows_coarse_f[l] = torch.stack(flows_coarse_f[l], 0)
    #                 occs_coarse_f[l] = torch.stack(occs_coarse_f[l], 0)
    #         else:
    #             for l in range(len(flows_f)):
    #                 flows_f[l] = flows_f[l][0]
    #                 occs_f[l] = occs_f[l][0]
    #             for l in range(len(flows_coarse_f)):
    #                 flows_coarse_f[l] = flows_coarse_f[l][0]
    #                 occs_coarse_f[l] = occs_coarse_f[l][0]
    #         output_dict['flow'] = flows_f
    #         output_dict['occ'] = occs_f
    #         output_dict['flow_coarse'] = flows_coarse_f
    #         output_dict['occ_coarse'] = occs_coarse_f
    #         return output_dict
    #     else:
    #         output_dict_eval = {}
    #         if len(list_imgs) > 2:
    #             out_flow = []
    #             out_occ = []
    #             for i in range(len(flows_f[0])):
    #                 out_flow.append(upsample2d_as(flows_f[-1][i], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow))
    #                 out_occ.append(upsample2d_as(occs_f[-1][i], list_imgs[0], mode="bilinear"))
    #             out_flow = torch.stack(out_flow, 0)
    #             out_occ = torch.stack(out_occ, 0)
    #         else:
    #             out_flow = upsample2d_as(flows_f[-1][0], list_imgs[0], mode="bilinear") * (1.0 / self._div_flow)
    #             out_occ = upsample2d_as(occs_f[-1][0], list_imgs[0], mode="bilinear")
    #         output_dict_eval['flow'] = out_flow
    #         output_dict_eval['occ'] = out_occ
    #         return output_dict_eval
