# import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

# from utils.utils import coords_grid, bilinear_sampler, upflow8
# from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from ..encoders import twins_svt_large_context, twins_svt_large
# from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
# from .twins import PosConv
from .encoder import MemoryEncoder, MemoryEncoder3Frames, MemoryEncoderNFrames
from .decoder import MemoryDecoder, MemoryDecoderOcclusion, MemoryDecoderConfidence, MemoryDecoder3Frames, MemoryDecoderNFrames
from .cnn import BasicEncoder

class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, network_input, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = network_input[0]
        # image2 = network_input[1]

        image1 = 2 * (network_input[:, 0] / 255.0) - 1.0
        image2 = 2 * (network_input[:, 1] / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        # return flow_predictions
        # TODO: currently the model outputs different flow_predictions formats for training and validation,
        #  here i care only about the flow so i will take care of that. change it later on
        occ, conf_x, conf_y = None, None, None
        if self.training:
            # return [-flow_predictions[-1]]
            return [flow_predictions[-1], occ, conf_x, conf_y]
        else:
            # return [-flow_predictions[0]]
            return [flow_predictions[0], occ, conf_x, conf_y]

    # def forward(self, image1, image2, output=None, flow_init=None):
    #     # Following https://github.com/princeton-vl/RAFT/
    #     image1 = 2 * (image1 / 255.0) - 1.0
    #     image2 = 2 * (image2 / 255.0) - 1.0
    #
    #     data = {}
    #
    #     if self.cfg.context_concat:
    #         context = self.context_encoder(torch.cat([image1, image2], dim=1))
    #     else:
    #         context = self.context_encoder(image1)
    #
    #     cost_memory = self.memory_encoder(image1, image2, data, context)
    #
    #     flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)
    #
    #     return flow_predictions


class FlowFormerOcclusion(nn.Module):
    def __init__(self, cfg):
        super(FlowFormerOcclusion, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoderOcclusion(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, network_input, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = network_input[0]
        # image2 = network_input[1]

        image1 = 2 * (network_input[:, 0] / 255.0) - 1.0
        image2 = 2 * (network_input[:, 1] / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            # shape in -> shape out : [B, 3, 512, 1024] -> [B, 256, 64, 128]
            context = self.context_encoder(image1)
        # [8192, 8, 128] = ([B, 3, 512, 1024], [B, 3, 512, 1024],  [8192, 1, 64, 128]   ,[B, 256, 64, 128])
        cost_memory = self.memory_encoder(image1, image2, data, context)
        # [B, 2 , 512, 1024]
        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)


        # return flow , occlusion
        conf_x, conf_y = None, None
        return [flow_predictions[0], flow_predictions[-1], conf_x, conf_y]


class FlowFormerConfidence(nn.Module):
    def __init__(self, cfg):
        super(FlowFormerConfidence, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoderConfidence(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, network_input, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = network_input[0]
        # image2 = network_input[1]

        #TODO: i normalize PPP and then send it here when it is no longer [0 .. 255]
        image1 = 2 * (network_input[:,0] / 255.0) - 1.0
        image2 = 2 * (network_input[:,1] / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            # shape in -> shape out : [B, 3, 512, 1024] -> [B, 256, 64, 128]
            context = self.context_encoder(image1)
        # [8192, 8, 128] = ([B, 3, 512, 1024], [B, 3, 512, 1024],  [8192, 1, 64, 128]   ,[B, 256, 64, 128])
        cost_memory = self.memory_encoder(image1, image2, data, context)
        # [B, 2 , 512, 1024]
        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)


        # return flow , occlusion
        # return [flow_predictions[0], flow_predictions[-1]]
        flow = flow_predictions[0]
        occ = None
        conf_x = flow_predictions[-1][:, 0:1]
        conf_y = flow_predictions[-1][:, 1:2]
        return [flow, occ, conf_x, conf_y]


class FlowFormer3Frames(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer3Frames, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder3Frames(cfg)
        self.memory_decoder = MemoryDecoder3Frames(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, network_input, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = network_input[0]
        # image2 = network_input[1]

        image1 = 2 * (network_input[:, 0] / 255.0) - 1.0
        image2 = 2 * (network_input[:, 1] / 255.0) - 1.0
        image3 = 2 * (network_input[:, 2] / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context12 = self.context_encoder(torch.cat([image1, image2], dim=1))
            context23 = self.context_encoder(torch.cat([image2, image3], dim=1))
            context13 = self.context_encoder(torch.cat([image1, image3], dim=1))
            cost_memory = self.memory_encoder(image1, image2, image3, data, torch.cat([context12, context23, context13], dim=1))
        else:
            context1 = self.context_encoder(image1)
            context2 = self.context_encoder(image2)
            context = torch.cat([context1, context2], dim=0)
            cost_memory = self.memory_encoder(image1, image2, image3, data, context)

        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init)

        # return flow_predictions
        # TODO: currently the model outputs different flow_predictions formats for training and validation,
        #  here i care only about the flow so i will take care of that. change it later on
        occ, conf_x, conf_y = None, None, None
        if self.training:
            # return [-flow_predictions[-1]]
            return [flow_predictions[-1], occ, conf_x, conf_y]
        else:
            # return [-flow_predictions[0]]
            return [flow_predictions[0], occ, conf_x, conf_y]


class FlowFormerNFrames(nn.Module):
    def __init__(self, cfg, N):
        super(FlowFormerNFrames, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoderNFrames(cfg, N)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, network_input, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        # image1 = network_input[0]
        # image2 = network_input[1]
        images = []
        for image in network_input[0]:
            im = 2 * (image / 255.0) - 1.0
            images.append(im.unsqueeze(0))

        data = {}

        if self.cfg.context_concat:
            contexts = []
            for im1, im2 in enumerate(zip(images, images[1:])):
                context = self.context_encoder(torch.cat([im1, im2], dim=1))
                contexts.append(context)

            images = torch.cat([image for image in images], dim=0).unsqueeze(0)
            cost_memory = self.memory_encoder(images, data, contexts)
        else:
            # contexts = []
            # for im in images:
            #     context = self.context_encoder(im)
            #     contexts.append(context)
            contexts = [self.context_encoder(images[0])]

            images = torch.cat([image for image in images], dim=0).unsqueeze(0)
            cost_memory = self.memory_encoder(images, data, contexts)

        contexts = contexts[0] # todo: i assumed we went to else statement earlier, find out what happens if we didnt
        flow_predictions = self.memory_decoder(cost_memory, contexts, data, flow_init=flow_init)
        """
        cost_memory.shape
        torch.Size([6912, 8, 128])
        contexts.shape
        torch.Size([1, 256, 54, 128])
        data['cost_maps'].shape
        torch.Size([6912, 1, 54, 128])
        """
        # return flow_predictions
        # TODO: currently the model outputs different flow_predictions formats for training and validation,
        #  here i care only about the flow so i will take care of that. change it later on
        occ, conf_x, conf_y = None, None, None
        if self.training:
            # return [-flow_predictions[-1]]
            return [flow_predictions[-1], occ, conf_x, conf_y]
        else:
            # return [-flow_predictions[0]]
            return [flow_predictions[0], occ, conf_x, conf_y]