import os.path
import warnings
from typing import Union, List, Dict

import numpy as np
import torch
from easydict import EasyDict

from RDND_proper.models.FlowNetPytorch.models.FlowNet1S import FlowNet1S
from temporary_utils import load_model_from_checkpoint
from torch import nn
from torchvision import transforms
from wrappers.default_checkpoints_dict import get_optical_flow_default_checkpoints_dict

from _internal_utils.parameter_checking import type_check_parameters
from _internal_utils.test_exceptions import raise_if_not
from _internal_utils.torch_utils import construct_tensor


class OpticalFlowWrapper(nn.Module):
    name = None

    def __init__(self):
        super().__init__()

    def forward(self,
                frames: Union[torch.Tensor, np.array, tuple, list],
                reference_frame: str = 'next') -> List[Dict]:
        # TODO format inputs and outputs properly
        frames, reference_frame = self._format_parameters_forward(frames=frames, reference_frame=reference_frame)
        results = []

        if reference_frame == 'next':
            for i in range(len(frames) - 1):
                result = self._get_optical_flow(frames[i], frames[i + 1])
                results.append(result)

        elif reference_frame == 'center':
            center_frame = frames[int(len(frames) / 2)]
            for i in range(len(frames)):
                result = self._get_optical_flow(frames[i], center_frame)
                results.append(result)

        # TODO get some consultant about the output - if layer, what should the output look like?
        #  I currently made a list of dicts but I think I'll make it dict of lists / tensors. Need to think
        return results

    def _get_optical_flow(self,
                          frame1: torch.Tensor,
                          frame2: torch.Tensor) -> Dict:
        pass

    def _format_parameters_forward(self,
                                   frames: Union[torch.Tensor, np.array, tuple, list],
                                   reference_frame: str) -> (torch.Tensor, str):
        type_check_parameters([(frames, (torch.Tensor, np.ndarray, tuple, list)), (reference_frame, str)])
        frames = construct_tensor(frames)  # TODO make device choice clever
        frames = torch.transpose(frames, 3, 1)
        frames = torch.transpose(frames, 2, 3)  # TODO make this transpose built in at Anvil
        valid_reference_methods = ['next', 'center']
        raise_if_not(reference_frame in valid_reference_methods, message="Invalid comparison method for optical flow")

        return frames, reference_frame


class NeuralOpticalFlowWrapper(OpticalFlowWrapper):
    def __init__(self, model=None, checkpoint_path=None):
        super().__init__()
        self.model = model

        self._initialize_model(checkpoint_path)
        self.train_dict = None

    def _initialize_model(self, checkpoint_path: str = None) -> None:
        # Get network weights for initialization
        if not checkpoint_path:
            default_optical_flow_checkpoints = get_optical_flow_default_checkpoints_dict()
            if default_optical_flow_checkpoints[self.name]:
                warnings.warn("Using default weights for {}".format(self.name))
                checkpoint_path = default_optical_flow_checkpoints[self.name]
            else:
                raise RuntimeError("No default weights stored for the chosen method. Please send checkpoint path")

        # create training dictionary
        self.train_dict = EasyDict(
            models_folder=os.path.dirname(checkpoint_path),
            Network_checkpoint_step=0,
            flag_remove_dataparallel_wrapper_from_checkpoint=False,
            load_Network_filename=checkpoint_path
        )

        # Initialize model
        self.model, self.train_dict = load_model_from_checkpoint(self.train_dict, self.model)


class FlowNet1SimpleWrapper(NeuralOpticalFlowWrapper):
    name = 'FlowNet1Simple'

    def __init__(self, checkpoint_path: str = None):
        super().__init__(model=FlowNet1S(), checkpoint_path=checkpoint_path)

    def _get_optical_flow(self,
                          frame1: torch.Tensor,
                          frame2: torch.Tensor) -> Dict:
        """
        input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])
    array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()
        :param frame1:
        :param frame2:
        :return:
        """
        model_input = torch.cat((frame1, frame2)).unsqueeze(0)
        input_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        ])
        model_input = input_transform(model_input)

        optical_flow_outputs = EasyDict(
            flow=None
        )
        self.model.eval()
        with torch.no_grad():
            optical_flow_outputs.flow = self.model(model_input)[0]

        return optical_flow_outputs
