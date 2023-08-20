import numpy as np
import pyransac3d
from skimage.measure import LineModelND, ransac

from application.pipeline.AlgorithmParams import AlgorithmParams


class RansacResults:
    def __init__(self, slopeX, slopeY, slopeZ, interceptX, interceptY, interceptZ, inliers, outliers):
        self.slopeX = slopeX
        self.slopeY = slopeY
        self.slopeZ = slopeZ
        self.interceptX = interceptX
        self.interceptY = interceptY
        self.interceptZ = interceptZ
        self.inliers = inliers
        self.outliers = outliers


def run_ransac(points: np.array, params: AlgorithmParams) -> RansacResults:
    # robustly fit line only using inlier data with RANSAC algorithm
    # inliers is boolean array
    line, inliers_bool = ransac(points, LineModelND, min_samples=params.r_min_samples,
                                   residual_threshold=params.r_residual_threshold, max_trials=params.r_max_iterations)
    outliers_bool = inliers_bool == False
    # line.params is an array of two arrays, the first being origin vector, and the second direction vector, in X, Y, Z
    intercepts = line.params[0]
    slopes = line.params[1]
    inliers = points[inliers_bool]
    outliers = points[outliers_bool]
    return RansacResults(slopes[2], slopes[1], slopes[0], intercepts[2], intercepts[1], intercepts[0], inliers, outliers)


# deprecated
def run_old_ransac(points: np.array, params: AlgorithmParams) -> RansacResults:
    line_detector = pyransac3d.Line()

    slopes, intercepts, inliers = line_detector.fit(points, thresh=params.r_radius_threshold,
                                                    maxIteration=params.r_max_iterations)
    return RansacResults(slopes, intercepts, inliers, points)
