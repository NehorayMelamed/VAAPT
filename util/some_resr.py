import torch
import numpy as np
number=3
for i in range(3,9):
    tensor_pt = torch.load(f"/home/nehoray/PycharmProjects/Shaback/Grounded-Segment-Anything/outputs{i}/segmentation.pt")

    numpy_array = tensor_pt.numpy()
    np.save(f'outputs{i}.npy', numpy_array)