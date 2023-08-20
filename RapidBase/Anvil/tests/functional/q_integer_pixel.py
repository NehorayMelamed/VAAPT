import cv2, torch

from torch import Tensor

from transforms import shift_matrix_integer_pixels
from tests.testing_utils.matrix_creations import tshowm


def shift_croc_stack_integer():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = torch.stack([Tensor(croc).permute((2, 0, 1)) for _ in range(5)], dim=0)
    shift_H = [100,0,40,0,100]
    shift_W = [100,100,100,100,100]
    croc_back = shift_matrix_integer_pixels(croc_input, shift_H, shift_W)
    both: Tensor = torch.zeros((H, W*2, C), dtype=torch.uint8)
    croc = Tensor(croc).to(torch.uint8)
    for i in range(5):
        both[:, :W, :] = croc
        both[:, W:, :] = croc_back[i].to(torch.uint8).permute(1, 2, 0)
        cv2.imshow("croc", both.numpy())
        cv2.waitKey(400)
    #cv2.imshow('croc', croc_back.to(torch.uint8).numpy().reshape(H, W, C))


def shift_croc_integer():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_back = shift_matrix_integer_pixels(croc_input, 100, 100).permute(1, 2, 0)
    both: Tensor = torch.zeros((H,W*2,C), dtype=torch.uint8)

    both[:, :W, :] = Tensor(croc).to(torch.uint8)
    both[:, W:, :] = croc_back.to(torch.uint8)
    cv2.imshow("croc", both.numpy())
    cv2.waitKey(0)
    #cv2.imshow('croc', croc_back.to(torch.uint8).numpy().reshape(H, W, C))


def shift_croc_integer_gray():
    #PASSED
    croc = cv2.imread('data/eye.jpg', cv2.IMREAD_GRAYSCALE)
    H, W = croc.shape[0], croc.shape[1]
    croc = croc.reshape((H, W))
    croc_back = shift_matrix_integer_pixels(croc, 100, 100)
    both: Tensor = torch.zeros((H, W*2), dtype=torch.uint8)
    both[:, :W] = Tensor(croc).to(torch.uint8)
    both[:, W:] = croc_back.to(torch.uint8)
    cv2.imshow("croc", both.numpy())
    cv2.waitKey(0)
    #cv2.imshow('croc', croc_back.to(torch.uint8).numpy().reshape(H, W, C))


def now():
    a = torch.load("alligator.pt")
    a = torch.stack([a,a,a])
    b = shift_matrix_integer_pixels(tuple(a), Tensor([10,100.0,20]), 200)
    tshowm([a[0],b[0],b[1]], 'o')


if __name__ == '__main__':
    now()
