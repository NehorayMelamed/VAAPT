import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
from RDND_proper.models.RLSP.RLSP.pytorch.parameters import params
#from RDND_proper.models.RLSP.RLSP.pytorch.functions import shuffle_down, shuffle_up


# RLSP architecture with RGB output (Y output in the paper)
class RLSP(nn.Module):



    def __init__(self):
        super(RLSP, self).__init__()

        # define / retrieve model parameters
        factor = params["factor"]
        filters = params["filters"]
        kernel_size = params["kernel size"]
        layers = params["layers"]
        state_dim = params["state dimension"]

        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(3*3 + 3*factor**2 + state_dim, filters, kernel_size, padding=int(kernel_size/2))
        self.conv_list = nn.ModuleList([nn.Conv2d(filters, filters, kernel_size, padding=int(kernel_size/2)) for _ in range(layers-2)])
        self.conv_out = nn.Conv2d(filters, 3*factor**2 + state_dim, kernel_size, padding=int(kernel_size/2))


    def cell(self, x, fb, state):

        # retrieve parameters
        factor = params["factor"]

        # define network
        res = x[:, 1]  # keep x for residual connection

        input = torch.cat([x[:, 0], x[:, 1], x[:, 2],
                           shuffle_down(fb, factor),
                           state], -3)

        # first convolution                   
        x = self.act(self.conv1(input))

        # main convolution block
        for layer in self.conv_list:
                x = self.act(layer(x))

        x = self.conv_out(x)
        
        out = shuffle_up(x[..., :3*factor**2, :, :] + res.repeat(1, factor**2, 1, 1), factor)
        state = self.act(x[..., 3*factor**2:, :, :])

        return out, state

    def forward(self, x):
        
        # retrieve device
        device = params["device"]

        # retrieve parameters
        factor = params["factor"]
        state_dimension = params["state dimension"]

        seq = []
        for i in range(x.shape[1]):                

            if i == 0:
                out = shuffle_up(torch.zeros_like(x[:, 0]).repeat(1, factor**2, 1, 1), factor)
                state = torch.zeros_like(x[:, 0, 0:1, ...]).repeat(1, state_dimension, 1, 1)

                out, state = self.cell(torch.cat([x[:, i:i+1], x[:, i:i+2]], 1).to(device), out, state)

            elif i == x.shape[1]-1:
                
                out, state = self.cell(torch.cat([x[:, i-1:i+1], x[:, i:i+1]], 1).to(device), out, state)

            else:
                
                out, state = self.cell(x[:, i-1:i+2], out, state)

            seq.append(out)

        seq = torch.stack(seq, 1)
        
        return seq


class RLSP_dudy(nn.Module):
    def __init__(self, params):
        super(RLSP_dudy, self).__init__()

        ### define / retrieve model parameters ###
        self.upsample_factor = params["factor"]
        self.number_of_filters = params["filters"]
        self.kernel_size = params["kernel size"]
        self.number_of_layers= params["layers"]
        self.state_dimension = params["state dimension"]
        self.number_of_input_channels = params['number_of_input_channels']
        self.device = params["device"]
        
        ### Layers List: ###
        self.act = nn.ReLU()
        # self.act = nn.LeakyReLU()
        # self.act = nn.PReLU()
        self.conv1 = nn.Conv2d(3 * 3 + 3 * self.upsample_factor ** 2 + self.state_dimension, self.number_of_filters, self.kernel_size, padding=int(self.kernel_size / 2))
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.number_of_filters, self.number_of_filters, self.kernel_size, padding=int(self.kernel_size / 2)) for _ in range(self.number_of_layers- 2)])
        self.conv_out = nn.Conv2d(self.number_of_filters, 3 * self.upsample_factor ** 2 + self.state_dimension, self.kernel_size, padding=int(self.kernel_size / 2))
    def shuffle_down(self,x, factor):
        # format: (B, C, H, W)
        b, c, h, w = x.shape

        assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of " + str(factor) + "!"

        s = 1.5
        add_image = np.reshape(gf(x.cpu().detach().numpy(), sigma=[0, 0, s, s]),
                               [x.shape[0], factor ** 2, x.shape[2] // factor, x.shape[3] // factor])
        add_image = np.rint(np.clip(add_image, 0, 255)).astype(np.uint8)

        # he didnt sample down so 16 -> 48 and 153 -> 185

        """
        n = x.reshape(b, c, int(h/factor), factor, int(w/factor), factor)
        n = n.permute(0, 3, 5, 1, 2, 4)
        n = n.reshape(b, c*factor**2, int(h/factor), int(w/factor))
        """
        return torch.from_numpy(add_image).cuda()

    # RGB -> Y transform
    def myrgb2y(self,im):
        constants = torch.tensor([[65.738, 129.057, 25.064]]).cuda()
        # print(im.permute(1,2,0).shape)
        # print(constants.shape)
        mul = torch.matmul(im.permute(1, 2, 0), constants)
        # print(mul.shape)
        return torch.sum(mul / 256, 2) + 16

    def shuffle_up(self,x, factor):
        # format: (B, C, H, W)
        b, c, h, w = x.shape

        assert c % factor ** 2 == 0, "C must be a multiple of " + str(factor ** 2) + "!"

        # n = x.reshape(b, factor, factor, int(c/(factor**2)), h, w)
        # n = n.permute(0, 3, 4, 1, 5, 2)
        n = x.reshape(b, int(c / (factor ** 2)), factor * h, factor * w)

        return n


    def cell(self, x, fb, state):
        """
                        ### Keep x for Residual Connection: ###
                        res = x[:, 1]

                        ### Concatenate All Inputs Together: ###
                        input = torch.cat([x[:, 0], x[:, 1], x[:, 2],
                                           self.shuffle_down(fb, self.upsample_factor),
                                           state], -3)



                        ### Forward Concatenated Tensor Into Layers List: ###
                        x = self.act(self.conv1(input))
                        for layer in self.conv_list[:-1]:
                            x = self.act(layer(x))
                        x = self.conv_out(x)

                        ### Take First Part Of Output, Shuffle It To Reach High Resolution, Add Residual Branch To Get Final Output: ###
                        out = shuffle_up(x[..., :self.number_of_input_channels * self.upsample_factor ** 2, :, :] + res.repeat(1, self.upsample_factor ** 2, 1, 1), self.upsample_factor)

                        ### Take Second Part Of Output (after activation) As Hidden State For Next Run: ###
                        state = self.act(x[..., self.number_of_input_channels * self.upsample_factor ** 2:, :, :])

                        return out, state
        """

        layer = torch.cat(torch.unbind(x, axis=1), axis=1)

        input = torch.cat([layer, state, self.shuffle_down(fb, self.upsample_factor)], dim=1)

        # first convolution
        x = self.conv1(input.cuda())
        x = self.act(x)


        for layer in self.conv_list[:-1]:
            x = self.act(layer(x))

        # extra layer just for the next state
        state=self.conv_list[-1](x)

        x = self.conv_out(x)
        res_add = self.myrgb2y(x[:,1])
        x+=res_add

        out=self.shuffle_up(x,self.upsample_factor)

        return out, state



    def forward(self, x):
        seq = []

        ### Loop Over Input Frames: ###
        for i in range(x.shape[1]):

            ### Take Current 3 Consecutive Frames, Previous State and Previous Output: ###
            if i == 0:
                ### Initialize previous output and previous state with zeros at the beginning: ###
                out = self.shuffle_up(torch.zeros_like(x[:, 0]).repeat(1, self.upsample_factor ** 2, 1, 1), self.upsample_factor)
                state = torch.zeros_like(x[:, 0, 0:1, ...]).repeat(1, self.state_dimension, 1, 1)
                current_frames_input = torch.cat([x[:, i:i + 1], x[:, i:i + 2]], 1).to(self.device)  #[frame0,frame0,frame1]
            elif i == x.shape[1] - 1:
                current_frames_input = torch.cat([x[:, i - 1:i + 1], x[:, i:i + 1]], 1).to(self.device)  #[frame(N-1),frame(N),frame(N)]
            else:
                current_frames_input = x[:, i - 1:i + 2] #[frame(I-1), frame(I), frame(I+1)]

            ### Pass Inputs Into Cell: ###
            out, state = self.cell(current_frames_input, out, state)

            ### Append Current Output To Sequence List: ###
            seq.append(out)

        ### Take Sequence List And Make It A Long Tensor: ###
        seq = torch.stack(seq, 1)
        seq = seq[:,-1]  #get last frame
        # seq = seq[len(seq)//2+1]
        return seq


class RLSP_dudy_single_run(nn.Module):
    def __init__(self, params):
        super(RLSP_dudy_single_run, self).__init__()

    def forward(self, x):
        ### Assign The Different Outputs: ###
        current_frames_input = x[0]
        out = x[1]
        state = x[2]

        ### Pass Inputs Into Cell: ###
        out, state = self.cell(current_frames_input, out, state)

        return (out, state)

from RapidBase.Utils.MISCELENEOUS import RGB2Gray
class RLSP_Recursive(nn.Module):
    def __init__(self, params):
        super(RLSP_Recursive, self).__init__()

        ### define / retrieve model parameters ###
        self.upsample_factor = params["factor"]
        self.number_of_filters = params["filters"]
        self.kernel_size = params["kernel size"]
        self.number_of_layers = params["layers"]
        self.state_dimension = params["state dimension"]
        self.number_of_input_channels = params['number_of_input_channels']
        self.device = params["device"]

        ### Layers List: ###
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU()
        #self.conv1 = nn.Conv2d(3 * 3 + 3 * self.upsample_factor ** 2 + self.state_dimension, self.number_of_filters,
        #                       self.kernel_size, padding=int(self.kernel_size / 2))#153
        #146
        self.conv1=nn.Conv2d(140, self.number_of_filters,
                               self.kernel_size, padding=int(self.kernel_size / 2))




        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.number_of_filters, self.number_of_filters, self.kernel_size,
                       padding=int(self.kernel_size / 2)) for _ in range(self.number_of_layers - 2)])

        self.conv_out = nn.Conv2d(self.number_of_filters, 16*3,
                                  self.kernel_size, padding=int(self.kernel_size / 2))

        #3 * self.upsample_factor ** 2 + self.state_dimension
        ### Hidden Recurrent State: ###
        self.out = None
        self.state = None

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv_out.weight)
        for layer in self.conv_list:
            nn.init.xavier_uniform_(layer.weight)


    def shuffle_down(self, x, factor):
        # format: (B, C, H, W)
        #print(x.shape)
        b, c, h, w = x.shape

        assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of " + str(factor) + "!"

        s = 1.5
        add_image = np.reshape(gf(x.cpu().detach().numpy(), sigma=[0, 0, s, s])[:,:,0::4,0::4],
                               [x.shape[0], c, x.shape[2] // factor, x.shape[3] // factor])
        add_image = np.rint(np.clip(add_image, 0, 255)).astype(np.uint8)

        # he didnt sample down so 16 -> 48 and 153 -> 185

        """
        n = x.reshape(b, c, int(h/factor), factor, int(w/factor), factor)
        n = n.permute(0, 3, 5, 1, 2, 4)
        n = n.reshape(b, c*factor**2, int(h/factor), int(w/factor))
        """

        return torch.from_numpy(add_image).cuda()

        # RGB -> Y transform

    # def myrgb2y(self, im):
    #     constants = torch.tensor([[65.738, 129.057, 25.064]]).cuda()
    #     #print(im.shape)
    #     #print(constants.shape)
    #     mul = torch.matmul(im[:,None].permute(1, 2,3, 0), constants)
    #     # print(mul.shape)
    #     return torch.sum(mul / 256, 2) #+ 16

    def myrgb2y2(self, im):
        constants = torch.tensor([[65.738, 129.057, 25.064]]).cuda()
        #print(im.shape)
        # print(constants.shape)
        mul = im.permute(0, 2,3,1)* constants
        # print(mul.shape)
        mul = mul.permute(0,3,1,2)
        summation=torch.sum(mul / 256, 1)# + 16

        return summation.reshape(
            [
                summation.shape[0],
                1,
                summation.shape[1],
                summation.shape[2]

            ]

        )

    def shuffle_up(self, x, factor):
        # format: (B, C, H, W)
        b, c, h, w = x.shape

        assert c % factor ** 2 == 0, "C must be a multiple of " + str(factor ** 2) + "!"

        # n = x.reshape(b, factor, factor, int(c/(factor**2)), h, w)
        # n = n.permute(0, 3, 4, 1, 5, 2)
        n = x.reshape(b, int(c / (factor ** 2)), factor * h, factor * w)

        return n

    def cell(self, x, fb, state):
        """
        ### Keep x (center frame) for Residual Connection: ###
        res = x[:, 1]

        ### Concatenate All Inputs Together: ###
        input = torch.cat([x[:, 0], x[:, 1], x[:, 2],
                           shuffle_down(fb, self.upsample_factor),
                           state], -3)

        ### Forward Concatenated Tensor Into Layers List: ###
        x = self.act(self.conv1(input))
        for layer in self.conv_list:
            x = self.act(layer(x))
        x = self.conv_out(x)

        ### Take First Part Of Output, Shuffle It To Reach High Resolution, Add Residual Branch To Get Final Output (which is low resolution, repeated and shuffled up): ###
        out = shuffle_up(x[..., :self.number_of_input_channels * self.upsample_factor ** 2, :, :] + res.repeat(1,self.upsample_factor ** 2,1, 1), self.upsample_factor)

        ### Take Second Part Of Output (after activation) As Hidden State For Next Run: ###
        state = self.act(x[..., self.number_of_input_channels * self.upsample_factor ** 2:, :, :])

        return out, state
        """
        # res_add = self.myrgb2y2(x[:, 1, :, :, :])
        res_add = RGB2Gray(x[:,1,:,:,:] + 0)
        layer = torch.cat(torch.unbind(x, axis=1), axis=1)

        input = torch.cat([layer, state, self.shuffle_down(fb, self.upsample_factor)], dim=1)

        # first convolution
        x = self.conv1(input.cuda())
        x = self.act(x)

        for layer in self.conv_list[:-1]:
            x = self.act(layer(x))

        # extra layer just for the next state
        state = self.conv_list[-1](x)
        #todo(aviram) add activation?
        state = self.act(state)

        x = self.conv_out(x)
        x += res_add

        out = nn.PixelShuffle(self.upsample_factor)(x)#self.shuffle_up(x, self.upsample_factor)

        return out, state

    def forward(self, x):

        ### Pass Inputs Into Cell: ###
        #for i in range(x.shape[1]):
        #    x[:,i]=self.myrgb2y(x[0,i])
        #x=self.myrgb2y2(x)

        self.out, self.state = self.cell(x, self.out, self.state)

        return self.out




