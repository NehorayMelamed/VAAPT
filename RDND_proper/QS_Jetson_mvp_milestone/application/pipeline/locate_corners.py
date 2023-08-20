import vpi
import numpy as np


class HarrisCorner:
    def __init__(self, row: int, column: int, score: float):
        self.row = row
        self.column = column
        self.score = score


def discover_corners(frame_stack: np.array, senstivivty = 0.0625,  strength = 20.0,  min_nms_distance = 8.0):
    all_corners = []
    frame_stack = frame_stack.astype(np.float32)
    with vpi.Backend.CUDA:
        for i in range(frame_stack.shape[0]):
            corners, scores = vpi.asimage(frame_stack[i], vpi.F32).harriscorners(sensitivity=senstivivty, strength = strength,
                                                                                 min_nms_distance=min_nms_distance)
            corners = corners.cpu()
            scores = scores.cpu()
            # append Harris corners for each frame, all corners = 2D list of dimensions Image X its corners
            all_corners.append([HarrisCorner(row=corners[i][1], column=corners[i][0], score=scores[i]) for i in range(scores.size)])
    return all_corners
