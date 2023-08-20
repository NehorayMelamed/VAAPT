import os
from os import path as osp
import shutil

root_dir = '/raid/Pytorch_Checkpoints'
dates = sorted(os.listdir(root_dir))

for date in dates:
    print ('started date ', date)
    date_dir = osp.join(root_dir, date)
    inference_dir = osp.join(date_dir, 'Inference')
    checkpoints_dir = osp.join(date_dir, 'Model_New_Checkpoints')

    for s in os.listdir(inference_dir): # Remove images (other than first and last)
        s_dir = osp.join(inference_dir,s)
        for folder in os.listdir(s_dir):
            imgs_folder = osp.join(s_dir,folder)
            steps = sorted(os.listdir(imgs_folder))
            for step in range(len(steps)-3):
                if step<3:
                    continue
                folder_to_delete = osp.join(imgs_folder,steps[step])
                shutil.rmtree(folder_to_delete)

    for s in os.listdir(checkpoints_dir): # Remove checkpoints (other than last)
        s_dir = osp.join(checkpoints_dir,s)
        checkpoints = sorted(os.listdir(s_dir))
        checkpoints.remove('network_description.txt')
        checkpoints_sorted = sorted(checkpoints, key=lambda cp: int(cp.split('Step')[1].split('.')[0]))
        for cp in range(len(checkpoints_sorted)-3):
            cp_path = osp.join(s_dir,checkpoints_sorted[cp])
            os.remove(cp_path)

    print('finished date ', date)