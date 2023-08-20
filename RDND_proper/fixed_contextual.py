import torch
import torch.nn as nn

from contextual_loss.modules.vgg import VGG19
from contextual_loss import functional as F
from contextual_loss .config import LOSS_TYPES

class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = False,
                 vgg_layer: str = 'relu3_4',
                 device: int = 0):

        super(ContextualLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES,\
            f'select a loss type from {LOSS_TYPES}.'

        self.band_width = band_width
        self.device = device

        if use_vgg:
            self.vgg_model = VGG19().to(self.device)
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

            self.vgg_mean = self.vgg_mean.to(self.device)
            self.vgg_std = self.vgg_std.to(self.device)

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean).div(self.vgg_std)
            y = y.sub(self.vgg_mean).div(self.vgg_std)

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        return F.contextual_loss(x, y, self.band_width)


import time
device = 'cpu'
device = 1
x = torch.randn(6,3,256,256).to(device)
y = torch.randn(6,3,256,256).to(device)

crit = ContextualLoss(use_vgg=True, vgg_layer='relu5_4', device=device)
start_time = time.time()
loss = crit(x,y)
print (3)
print (loss)
print("--- %s seconds ---" % (time.time() - start_time))


