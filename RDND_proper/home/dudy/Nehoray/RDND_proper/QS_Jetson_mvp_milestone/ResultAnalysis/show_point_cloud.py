import sys, os
import open3d as o3d
from typing import List, Dict, Tuple
import pyransac3d as pyrsc
import numpy as np


def exit_on(msg: str, code=-1):
    print(msg)
    sys.exit(code)


def check_input(argv: List[str]) -> Tuple[str, str]:
    # returns data dir, video_name
    argc = len(argv)
    if argc != 3:
        exit_on("Usage: python3 show_point_cloud.py [data_dir] [video_prefix]")
    if not os.path.exists(argv[1]):
        exit_on("Given data directory does not exist")
    s = os.path.sep
    pa = f"{argv[1]}results{s}RANSAC{s}REGRESSIONS{s}{argv[2]}.csv"
    if not os.path.exists(pa):
        print(pa)
        exit_on("Given Video Prefix data directory combination does not exist")
    return argv[1], argv[2]


class Regression:
    def __init__(self, slopes: np.array, intercepts: np.array):
        self.slopes = slopes
        self.intercepts = intercepts


def extract_line(tokenized_line: List[str]) -> Regression:
    slopes = np.array([0, 0, 0], dtype=np.float32)
    intercepts = np.array([0, 0, 0], dtype=np.float32)
    for i in range(2):
        slopes[i] = int(round(float(tokenized_line[i+1])))
        intercepts[i] = int(round(float(tokenized_line[i+3])))
    slopes[2] = 1
    intercepts[2] = 0
    return Regression(slopes=slopes, intercepts=intercepts)


class Visualizer:
    def __init__(self, data_folder: str, video_name: str):
        s = os.path.sep
        self.data_folder = data_folder
        self.video_name = video_name
        self.regression_lines: Dict[int, Regression] = dict()
        with open(f"{data_folder}results{s}RANSAC{s}REGRESSIONS{s}{video_name}.csv", 'r') as regressions:
            regression_lines = regressions.readlines()[1:]
            for line in regression_lines:
                tokenized_line = line.split(',')
                self.regression_lines[int(round(float(tokenized_line[0])))] = extract_line(tokenized_line)

    def show_all_clouds(self, data_folder: str):
        for i in sorted(self.regression_lines.keys()):
            self.show_nth_cloud(i)

    def get_nth_line(self, n: int, radius=1, height=200):
        regressed_line = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        regressed_line.compute_vertex_normals()
        regressed_line.paint_uniform_color([1, 0, 0])
        regression_line = self.regression_lines[n]
        rotation_matrix = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], regression_line.slopes)
        regressed_line = regressed_line.rotate(rotation_matrix, center=[0, 0, 0])
        regressed_line = regressed_line.translate(tuple(regression_line.intercepts))
        return regressed_line

    def show_nth_cloud(self, n: int):
        s = os.path.sep
        try:
            outliers = np.load(f"{self.data_folder}results{s}RANSAC{s}OUTLIERS{s}{self.video_name}_{n}.npy")
            outliers_pcd = o3d.geometry.PointCloud()
            outliers_pcd.points = o3d.utility.Vector3dVector(outliers)
            outliers_pcd.paint_uniform_color([0.2, 0.2, 0.8])
            inliers = np.load(f"{self.data_folder}results{s}RANSAC{s}INLIERS{s}{self.video_name}_{n}.npy")
            inliers_pcd = o3d.geometry.PointCloud()
            inliers_pcd.points = o3d.utility.Vector3dVector(inliers)
            inliers_pcd.paint_uniform_color([0.8, 0.2, 0.2])
            regressed_line = self.get_nth_line(n)
            o3d.visualization.draw_geometries([regressed_line, outliers_pcd, inliers_pcd])

        except FileNotFoundError:
            print("Given point cloud does not exist")
        except KeyError:
            print("Given Line Does not have any Points")


if __name__ == '__main__':
    data_dir, video_name = check_input(sys.argv)
    painter = Visualizer(data_dir, video_name)
    while True:
        pc_num = input("Which Point Cloud Should Be Shown? >>> ")
        if pc_num.strip().upper() == "ALL":
            painter.show_all_clouds(data_dir)
        else:
            try:
                painter.show_nth_cloud(int(pc_num))
            except ValueError: # entered number wasn't a number or all
                print("Please enter a valid number or ALL")
                continue
