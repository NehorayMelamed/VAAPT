import math

import torch
from torch import nn, Tensor


def gaussian_window(N: int, std: float) -> Tensor:
    folded_window = torch.zeros(N//2)
    for i in range(N//2):
        ex = (-1 / 2) * (((i+1) / std) ** 2)
        folded_window[i] = math.exp(ex)
    if N % 2 == 0: # symetricality of window
        window = torch.cat([folded_window.flip(dims=[0]), folded_window])
    else:
        window = torch.cat([folded_window.flip(dims=[0]), Tensor([1]), folded_window])
    return window / window.max()


class canny_edge_detection(nn.Module):
    def __init__(self, threshold=10.0):
        super(canny_edge_detection, self).__init__()

        self.threshold = threshold

        filter_size = 5
        generated_filters = gaussian_window(filter_size, std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(generated_filters)
        self.gaussian_filter_horizontal.bias.data.copy_(Tensor([0.0]))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(generated_filters.T)
        self.gaussian_filter_vertical.bias.data.copy_(Tensor([0.0]))


        sobel_filter = Tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(sobel_filter)
        self.sobel_filter_horizontal.bias.data.copy_(Tensor([0.0]))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(sobel_filter.T)
        self.sobel_filter_vertical.bias.data.copy_(Tensor([0.0]))

        # filters were flipped manually
        filter_0 = Tensor([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = Tensor([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = Tensor([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = Tensor([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = Tensor([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = Tensor([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = Tensor([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = Tensor([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = torch.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(all_filters[:, None, ...])
        self.directional_filter.bias.data.copy_(torch.zeros(all_filters.shape[0],))

        self.gaussian_filter_horizontal = self.gaussian_filter_horizontal.to('cuda')
        self.gaussian_filter_vertical = self.gaussian_filter_vertical.to('cuda')
        self.sobel_filter_horizontal = self.sobel_filter_horizontal.to('cuda')
        self.sobel_filter_vertical = self.sobel_filter_vertical.to('cuda')
        self.directional_filter = self.directional_filter.to('cuda')

    def forward(self, img, threshold=None):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation = torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD
        if threshold is None:
            threshold = self.threshold

        thresholded = thin_edges.clone()
        thresholded[thin_edges<threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold
