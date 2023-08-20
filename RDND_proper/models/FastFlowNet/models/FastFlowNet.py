import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .correlation_package.correlation import Correlation
from RDND_proper.models.star_flow.models.correlation_package.correlation import CorrTorch as Correlation

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class Warp_Tensors_Layer(nn.Module):
    def __init__(self, *args):
        super(Warp_Tensors_Layer, self).__init__()
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.X0 = None
        self.Y0 = None
        self.grid = None
    def forward(self, input_tensors, flo):
        if self.grid is None:
            self.B, self.C, self.H, self.W = input_tensors.shape
            xx = torch.arange(0, self.W).view(1, -1).repeat(self.H, 1)
            yy = torch.arange(0, self.H).view(-1, 1).repeat(1, self.W)
            xx = xx.view(1, 1, self.H, self.W).repeat(self.B, 1, 1, 1)
            yy = yy.view(1, 1, self.H, self.W).repeat(self.B, 1, 1, 1)
            self.grid = torch.cat([xx, yy], 1).to(input_tensors.device)
        vgrid = self.grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(self.W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(self.H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  #TODO: probably takes a lot of time and is very wastefull....
        output = F.grid_sample(input_tensors, vgrid, mode='bilinear')
        return output

class FastFlowNet(nn.Module):
    def __init__(self, groups=3):
        super(FastFlowNet, self).__init__()

        self.warp_layer_1 = Warp_Tensors_Layer()
        self.warp_layer_2 = Warp_Tensors_Layer()
        self.warp_layer_3 = Warp_Tensors_Layer()
        self.warp_layer_4 = Warp_Tensors_Layer()
        self.warp_layer_5 = Warp_Tensors_Layer()

        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.correlation_layer = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.index = torch.tensor([0, 2, 4, 6, 8, 
                10, 12, 14, 16, 
                18, 20, 21, 22, 23, 24, 26, 
                28, 29, 30, 31, 32, 33, 34, 
                36, 38, 39, 40, 41, 42, 44, 
                46, 47, 48, 49, 50, 51, 52, 
                54, 56, 57, 58, 59, 60, 62, 
                64, 66, 68, 70, 
                72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)        
        output = F.grid_sample(x, vgrid, mode='bilinear')
        return output


    def forward(self, x):
        ### Get Input Images: ###
        img1 = x[0]
        img2 = x[1]
        # img1 = x[:, :3, :, :]
        # img2 = x[:, 3:6, :, :]
        
        ### Get 1st Stage Features: ###
        f1_1 = self.pconv1_2(self.pconv1_1(img1))
        f2_1 = self.pconv1_2(self.pconv1_1(img2))
        
        ### Get 2nd Stage Features: ###
        f1_2 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f1_1)))
        f2_2 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f2_1)))
        
        ### Get 3rd Stage Features: ###
        f1_3 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f1_2)))
        f2_3 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f2_2)))
        
        ### Next Stage Features Are Simply AvgPooling Of The Above: ###
        f1_4 = F.avg_pool2d(f1_3, kernel_size=(2, 2), stride=(2, 2))
        f2_4 = F.avg_pool2d(f2_3, kernel_size=(2, 2), stride=(2, 2))
        f1_5 = F.avg_pool2d(f1_4, kernel_size=(2, 2), stride=(2, 2))
        f2_5 = F.avg_pool2d(f2_4, kernel_size=(2, 2), stride=(2, 2))
        f1_6 = F.avg_pool2d(f1_5, kernel_size=(2, 2), stride=(2, 2))
        f2_6 = F.avg_pool2d(f2_5, kernel_size=(2, 2), stride=(2, 2))
        
        ### Start Using Features To Produce Correlations And From There (together with context features) Produce Flow In Pyramid Fashion: ###
        flow7_up = torch.zeros(f1_6.size(0), 2, f1_6.size(2), f1_6.size(3)).to(f1_5.device)
        correlation_matrix_6 = self.correlation_layer(f1_6, f2_6)
        cv6 = torch.index_select(correlation_matrix_6, dim=1, index=self.index.to(f1_6.device).long())
        r1_6 = self.rconv6(f1_6)
        cat6 = torch.cat([cv6, r1_6, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f2_5_w = self.warp_layer_1(f2_5, flow6_up*0.625)  #TODO: why these factors here?!!?!? because of training fudge factors?!?!?
        correlation_matrix_5 = self.correlation_layer(f1_5, f2_5_w)
        cv5 = torch.index_select(correlation_matrix_5, dim=1, index=self.index.to(f1_5).long())
        r1_5 = self.rconv5(f1_5)
        cat5 = torch.cat([cv5, r1_5, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f2_4_w = self.warp_layer_2(f2_4, flow5_up*1.25)
        correlation_matrix_4 = self.correlation_layer(f1_4, f2_4_w)
        cv4 = torch.index_select(correlation_matrix_4, dim=1, index=self.index.to(f1_4).long())
        r1_4 = self.rconv4(f1_4)
        cat4 = torch.cat([cv4, r1_4, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f2_3_w = self.warp_layer_3(f2_3, flow4_up*2.5)
        correlation_matrix_3 = self.correlation_layer(f1_3, f2_3_w)
        cv3 = torch.index_select(correlation_matrix_3, dim=1, index=self.index.to(f1_3).long())
        r1_3 = self.rconv3(f1_3)
        cat3 = torch.cat([cv3, r1_3, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f2_2_w = self.warp_layer_4(f2_2, flow3_up*5.0)
        correlation_matrix_2 = self.correlation_layer(f1_2, f2_2_w)
        cv2 = torch.index_select(correlation_matrix_2, dim=1, index=self.index.to(f1_2).long())
        r1_2 = self.rconv2(f1_2)
        cat2 = torch.cat([cv2, r1_2, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up
        
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2