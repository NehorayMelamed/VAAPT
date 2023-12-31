from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS

import numpy as np
import random

CRF_FOLDER = {
    -2:'X4_start1',
    -1:'X4_start1',
    0 :'X4_start1',
    15:'X4_crf15', 
    25:'X4_crf25',
    35:'X4_crf35'}


@DATASETS.register_module()
class SRREDSMultipleGTDataset(BaseSRDataset):
    """REDS dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 test_mode=False,
                 crf=-1):
        super().__init__(pipeline, scale, crf, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        
        if test_mode == False:
            self.lq_folder = self.lq_folder.replace('X4_start1', CRF_FOLDER[crf])

        txt = 'Test' if test_mode else 'Train'
        # print('mode: ', txt, '  ', self.lq_folder) 

        self.data_infos = self.load_annotations()

    def generate_crop_video_list(self, v_list):
        r_v_list = []
        
        for v in v_list:
            for i in range(2):
                for j in range(2):
                    v_name = "{}_{}_{}".format(v, i, j)
                    r_v_list.append(v_name)

        return r_v_list


    def load_annotations(self):
        """Load annoations for REDS dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # generate keys
        keys = [f'{i:03d}' for i in range(0, 270)]
        val_test_partion = ['240', '241', '246', '257', '000', '011', '015', '020']

        if self.val_partition == 'REDS4':
            val_partition = val_test_partion[-4:]
            val_partition = self.generate_crop_video_list(val_partition)

        elif self.val_partition == 'REDS4Val': 
            val_partition = val_test_partion[:4]
        elif self.val_partition == 'official':
            val_partition = [f'{i:03d}' for i in range(240, 270)]
        else:
            raise ValueError(
                f'Wrong validation partition {self.val_partition}.'
                f'Supported ones are ["official", "REDS4"]')
        # print(val_partition)
        if self.test_mode:
            keys = val_partition
        else:
            keys = [v for v in keys if v not in val_partition]
        # print(keys)
        data_infos = []
        for key in keys:
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=100,
                    num_input_frames=self.num_input_frames))

        return data_infos
