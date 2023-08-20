import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RDND_proper.models.RAFT.core.update import BasicUpdateBlock, SmallUpdateBlock
from RDND_proper.models.RAFT.core.extractor import BasicEncoder, SmallEncoder
from RDND_proper.models.RAFT.core.corr import CorrBlock, AlternateCorrBlock
from RDND_proper.models.RAFT.core.utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        ### Decide on dropout and alternate correlation implementation: ###
        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        ### Choose parameters according to whether to use small network or large: ###
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4 #TODO: understand this
            args.corr_radius = 3 #TODO: understand this
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)  #feature network
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)  #context network
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        
        ### Reshape & Softmax Mask Tensor For Correct (Efficient?) Multiplication For Convext Upsampling: ###
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        
        ### Effectively Multiply Flow To Be In Correct Form For Convext Upsampling: ###
        up_flow = F.unfold(8 * flow, [3,3], padding=1)  #[B,2,H,W] -> [B,18,H*W]
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)  #[B,18,H*W] -> [B,2,9,1,1,H,W]
        
        ### Multiply Flow With Mask And Reshape (Convext Upsampling): ###
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        ### Center Images Around Zero: ### #TODO: add this normalization possibility to preprocessing object
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        ### Run The Feature Network: ###
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        ### Run The Correlation Block: ###
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        ### Run The Context Network: ###
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        ### Initialize Flow (represented as the difference between coords1-coords): ###
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        ### Loop Over Iterations Predicting Flow: ###
        flow_predictions = []
        for itr in range(iters):
            ### Get Current Coordinates And Correlation Volume: ###
            coords1 = coords1.detach()
            corr = corr_fn(coords1) #index correlation volume

            ### Get Current Flow & Pass Through (Iterative) Flow Update Block: ###
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            ### Update Coordinates: ###
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            ### upsample Predictions: ###
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            ### Add To Flow Predictions: ###
            flow_predictions.append(flow_up)

        if test_mode:  #TODO: change to self.is_training or something
            return coords1 - coords0, flow_up
            
        return flow_predictions

class RAFT_dudy(RAFT):
    def __init__(self, args):
        super(RAFT_dudy, self).__init__(args)
        self.number_of_iterations = args.number_of_iterations

    def forward(self, network_input):
        """ Estimate optical flow between pair of frames """
        ### Unpack Input Variables: ###
        image1 = network_input[0]
        image2 = network_input[1]
        iters = network_input[2]
        flow_init = network_input[3]
        upsample = network_input[4]
        test_mode = network_input[5]

        ### Run The Feature Network: ###
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        ### Run The Correlation Block: ###
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        ### Run The Context Network: ###
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        ### Initialize Flow (represented as the difference between coords1-coords): ###
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        ### Loop Over Iterations Predicting Flow: ###
        flow_predictions = []
        for itr in range(iters):
            ### Get Current Coordinates And Correlation Volume: ###
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            ### Get Current Flow & Pass Through (Iterative) Flow Update Block: ###
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            ### Update Coordinates: ###
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            ### upsample Predictions: ###
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            ### Add To Flow Predictions: ###
            flow_predictions.append(flow_up)

        if test_mode:  # TODO: change to self.is_training or something
            return coords1 - coords0, flow_up

        return flow_predictions


