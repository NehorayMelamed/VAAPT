import cv2, torch

from torch import Tensor

from transforms import affine_warp_matrix


def blur_shift_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_input = torch.stack([croc_input, croc_input, croc_input])
    croc_back = affine_warp_matrix(croc_input, shifts_y=Tensor([100,200,300]), scales=[0.4,1.2, 1.7], warp_method='bilinear')[2].permute((1, 2, 0))
    both: Tensor = torch.zeros((H, W * 2, C), dtype=torch.uint8)

    both[:, :W, :] = Tensor(croc).to(torch.uint8)
    # both[:, W:, :] = CenterCrop((H,W))()
    croc_back = croc_back.to(torch.uint8)
    print(croc_back.numpy().shape)
    cv2.imshow("croc", croc_back.numpy())
    cv2.waitKey(0)


if __name__ == '__main__':
    blur_shift_croc()
