import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3),
        nn.PReLU(),
        nn.ReplicationPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.PReLU()
    )

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(in_channels, out_channels, 3),
#         nn.LeakyReLU(),
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(out_channels, out_channels, 3),
#         nn.LeakyReLU()
#     )

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(in_channels, out_channels, 3),
#         nn.ReLU(),
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(out_channels, out_channels, 3),
#         nn.ReLU()
#     )


# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(in_channels, out_channels, 3),
#         nn.BatchNorm2d(out_channels),
#         nn.LeakyReLU(),
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(out_channels, out_channels, 3),
#         nn.BatchNorm2d(out_channels),
#         nn.LeakyReLU()
#     )

class double_relu(nn.Module):
    def __init__(self, slope=0.05):
        super(double_relu, self).__init__()
        # self.slope = slope
        self.slope = nn.Parameter(torch.Tensor([slope]))

    def forward(self, x):
        x[x<0] = self.slope*x[x<0]
        x[x>1] = self.slope*x[x>1]
        return x

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(in_channels, out_channels, 3),
#         double_relu(),
#         nn.ReplicationPad2d(1),
#         nn.Conv2d(out_channels, out_channels, 3),
#         double_relu()
#     )


class residual_double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3),
        nn.LeakyReLU(),
        nn.ReplicationPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.LeakyReLU()
        )

        self.projector_block = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.projector_block(x) + self.conv_block(x)


class residual_double_conv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3),
        nn.LeakyReLU(),
        nn.ReplicationPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3),
        )
        self.Act = nn.LeakyReLU()
        self.projector_block = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.Act(self.projector_block(x) + self.conv_block(x))

class residual_double_conv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3),
        nn.PReLU(),
        nn.ReplicationPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3),
        )
        self.Act = nn.PReLU()
        self.projector_block = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.Act(self.projector_block(x) + self.conv_block(x))


class Unet_large(nn.Module):
    def __init__(self,in_channels,out_channels) :
        super().__init__()

        conv_module = residual_double_conv2

        self.down_layer1 = conv_module(in_channels=in_channels, out_channels=16)
        self.down_layer2 = conv_module(in_channels=16, out_channels=32)
        self.down_layer3 = conv_module(in_channels=32, out_channels=64)
        self.down_layer4 = conv_module(in_channels=64, out_channels=64)
        self.layer5 = conv_module(in_channels=64, out_channels=64)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_layer4 = conv_module(in_channels=64+64, out_channels=64)
        self.up_layer3 = conv_module(in_channels=64+64, out_channels=64)
        self.up_layer2 = conv_module(in_channels=64+32, out_channels=32)
        self.up_layer1 = conv_module(in_channels=32+16, out_channels=32)

        self.last_layer = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1)
        self.Prelu = nn.PReLU()
        self.double_relu = double_relu()
        self.norm = nn.BatchNorm2d(8)
    def forward(self,x) :

        conv1 = self.down_layer1(x)
        x = self.max_pool(conv1) # x/2 chan=32

        conv2 = self.down_layer2(x)
        x = self.max_pool(conv2) # x/4 chan=64

        conv3 = self.down_layer3(x)
        x = self.max_pool(conv3) # x/8 chan=128

        conv4 = self.down_layer4(x) #x/8
        x = self.max_pool(conv4) # x/16 chan=256

        x = self.layer5(x) # x/16 chan=512

        x = self.upsample(x) # x/8
        x = nn.Upsample(size=[conv4.size()[2],conv4.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv4], dim=1)  # x/8 chan
        x = self.up_layer4(x)

        x = nn.Upsample(size=[conv3.size()[2],conv3.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_layer3(x)

        x = nn.Upsample(size=[conv2.size()[2],conv2.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_layer2(x)

        x = nn.Upsample(size=[conv1.size()[2],conv1.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_layer1(x)

        # x = self.norm(x)
        out = self.last_layer (x)
        # out = out.clamp(0,1) #TODO: temp!!!!
        out = torch.sigmoid(out)
        # out = self.double_relu(out)
        # out = self.Prelu(out)
        return out


class Unet_larger_NeuralGimbaless(nn.Module):
    def __init__(self,in_channels,out_channels) :
        super().__init__()
        ### Get number of outputs per pixel: ###
        #(*). 2 = (delta_x, delta_y);  3 = (delta_x, delta_y, conf);  4 = (delta_x, delta_y, conf_x, conf_y)
        self.number_of_outputs_per_pixel = 2
        conv_module = residual_double_conv

        self.down_layer1 = conv_module(in_channels=in_channels, out_channels=32)
        self.down_layer2 = conv_module(in_channels=32, out_channels=32)
        self.down_layer3 = conv_module(in_channels=32, out_channels=64)
        self.down_layer4 = conv_module(in_channels=64, out_channels=64)
        self.layer5 = conv_module(in_channels=64, out_channels=64)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_layer4 = conv_module(in_channels=64+64, out_channels=64)
        self.up_layer3 = conv_module(in_channels=64+64, out_channels=64)
        self.up_layer2 = conv_module(in_channels=64+32, out_channels=32)
        self.up_layer1 = conv_module(in_channels=32+32, out_channels=32)

        self.last_layer = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1)
        self.Prelu = nn.PReLU()
        self.double_relu = double_relu()
        self.norm = nn.BatchNorm2d(8)
    def forward(self,x) :

        conv1 = self.down_layer1(x)
        x = self.max_pool(conv1) # x/2 chan=32

        conv2 = self.down_layer2(x)
        x = self.max_pool(conv2) # x/4 chan=64

        conv3 = self.down_layer3(x)
        x = self.max_pool(conv3) # x/8 chan=128

        conv4 = self.down_layer4(x) #x/8
        x = self.max_pool(conv4) # x/16 chan=256

        x = self.layer5(x) # x/16 chan=512

        x = self.upsample(x) # x/8
        x = nn.Upsample(size=[conv4.size()[2],conv4.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv4], dim=1)  # x/8 chan
        x = self.up_layer4(x)

        x = nn.Upsample(size=[conv3.size()[2],conv3.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_layer3(x)

        x = nn.Upsample(size=[conv2.size()[2],conv2.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_layer2(x)

        x = nn.Upsample(size=[conv1.size()[2],conv1.size()[3]], mode='bilinear', align_corners=True) (x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_layer1(x)

        # x = self.norm(x)
        out = self.last_layer(x) * 1
        # out = out.clamp(0,1) #TODO: temp!!!!
        # out = torch.sigmoid(out)
        # out = self.double_relu(out)
        # out = self.Prelu(out)


        return out




class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.down_layer1 = double_conv(in_channels=3, out_channels=8)
        self.down_layer3 = double_conv(in_channels=8, out_channels=28)
        self.layer5 = double_conv(in_channels=28, out_channels=28)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_layer3 = double_conv(in_channels=28 + 28, out_channels=28)
        self.up_layer1 = double_conv(in_channels=28 + 8, out_channels=8)

        self.last_layer = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # return x
        conv1 = self.down_layer1(x)
        conv2 = self.max_pool(conv1)  # x/2 chan=32
        x = self.max_pool(conv2)  # x/4 chan=64

        conv3 = self.down_layer3(x)
        x = self.max_pool(conv3)  # x/8 chan=128
        conv4 = self.max_pool(x)  # x/16 chan=256

        x = self.layer5(conv4)  # x/16 chan=512

        x = self.upsample(x)  # x/8
        x = nn.Upsample(size=[conv4.size()[2], conv4.size()[3]], mode='nearest')(x)
        x = nn.Upsample(size=[conv3.size()[2], conv3.size()[3]], mode='nearest')(x)

        x = torch.cat([x, conv3], dim=1)
        x = self.up_layer3(x)
        x = nn.Upsample(size=[conv2.size()[2], conv2.size()[3]], mode='nearest')(x)
        x = nn.Upsample(size=[conv1.size()[2], conv1.size()[3]], mode='nearest')(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_layer1(x)

        out = self.last_layer(x)
        return out


