from collections import defaultdict
import os
import numpy as np

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS


@DATASETS.register_module()
class SRVid4Dataset(BaseSRDataset):
    """Vid4 dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a dict containing paired data and other information.

    It reads Vid4 keys from the txt file.
    Each line contains:

        1. folder name;
        2. number of frames in this clip (in the same folder);
        3. image shape, seperated by a white space.

    Examples:

    ::

        calendar 40 (320,480,3)
        city 34 (320,480,3)

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{:08d}'.
        metric_average_mode (str): The way to compute the average metric.
            If 'clip', we first compute an average value for each clip, and
            then average the values from different clips. If 'all', we
            compute the average of all frames. Default: 'clip'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 num_input_frames,
                 pipeline,
                 scale,
                 filename_tmpl='{:08d}',
                 metric_average_mode='all',
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames}.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        # self.ann_file = str(ann_file)
        self.num_input_frames = num_input_frames
        self.filename_tmpl = filename_tmpl
        if metric_average_mode not in ['clip', 'all']:
            raise ValueError('metric_average_mode can only be "clip" or '
                             f'"all", but got {metric_average_mode}.')

        self.metric_average_mode = metric_average_mode
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
        """Load annoations for Vid4 dataset.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        keys = ["calendar","city","foliage","walk"]
        keys = self.generate_crop_video_list(keys)

        self.folders = {}
        data_infos = []

        for key in keys:
            files= os.listdir(os.path.join(self.gt_folder,key))
            frame_num = len(files)
            self.folders[key] = int(frame_num)
            data_infos.append(
                dict(
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=frame_num,  
                    num_input_frames=frame_num))
        return data_infos


    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.
        Args:
            results (list[tuple]): The output of forward_test() of the model.
        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_results = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_results[metric].append(val)
        for metric, val_list in eval_results.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        if self.metric_average_mode == 'clip':
            for metric, values in eval_results.items():
                start_idx = 0
                metric_avg = 0
                for _, num_img in self.folders.items():
                    end_idx = start_idx + num_img
                    folder_values = values[start_idx:end_idx]
                    metric_avg += np.mean(folder_values)
                    start_idx = end_idx

                eval_results[metric] = metric_avg / len(self.folders)
        else:
            eval_results = {
                metric: sum(values) / len(self)
                for metric, values in eval_results.items()
            }

        return eval_results
