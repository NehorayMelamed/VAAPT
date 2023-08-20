import time
from typing import List
import numpy as np
import cv2, os
import glob

from application.pipeline.AlgorithmParams import AlgorithmParams
from application.pipeline.run_ransac import run_ransac, RansacResults
from application.pipeline.locate_corners import discover_corners, HarrisCorner
from application.pipeline.Subtractor import Subtractor
from functional_utils import *

class MainPipeline:
    def __init__(self, params: AlgorithmParams):
        self.params = params
        self.subtractor_engine = Subtractor(params)
        self.batch = 0
        self.regressions_file_path = self.params.ransac_regression_dir + self.params.video_name +".csv"
        self.regression_header = "RANSAC_BATCH_NUM,SLOPEx,SLOPEy,SLOPEz,INTERCEPTx,INTERCEPTy,INTERCEPTz,EXPOSURE_TIME\n"
        self.regressions_file = open(self.regressions_file_path, 'w+')
        self.regressions_file.write(self.regression_header)

    def save_harris_bgs(self, harris_corners: List[List[HarrisCorner]], bgs_stack: np.array):
        pass

    def save_bgs(self, bgs_stack: np.array):
        for i in range(bgs_stack.shape[0]):
            cv2.imwrite(self.params.BGS_dir + self.params.video_name + f"_{self.batch}_{i}.png", bgs_stack[i])
    
    def save_empty_ransac_results(self):
        np.save(self.params.ransac_outliers_dir + self.params.video_name + f"_{self.batch}.npy", np.array([], dtype=np.uint8))
        np.save(self.params.ransac_inliers_dir + self.params.video_name + f"_{self.batch}.npy", np.array([], dtype=np.uint8))

    def save_ransac_results(self, ransac_results: RansacResults):
        np.save(self.params.ransac_outliers_dir + self.params.video_name + f"_{self.batch}.npy", ransac_results.outliers)
        np.save(self.params.ransac_inliers_dir + self.params.video_name + f"_{self.batch}.npy", ransac_results.inliers)
        self.regressions_file.write(f"{self.batch},{ransac_results.slopeX},{ransac_results.slopeY},{ransac_results.slopeZ},{ransac_results.interceptX},{ransac_results.interceptY},{ransac_results.interceptZ},{self.params.exposure_time}\n")

    def close(self):
        self.regressions_file.close()

    def reopen(self):
        self.regressions_file = open(self.regressions_file_path, 'a')

    def extract_harris_points(self, harris_corners: List[List[HarrisCorner]]) -> np.array:
        raise NotImplemented
    
    def extract_bgs_points(self, frame_stack: np.array) -> np.array:
        # returns (N, 3) array
        # start index is time index of first frame
        white_indices = np.where(frame_stack != 0)
        return np.stack((white_indices[0], white_indices[1], white_indices[2])).transpose()

    def save_originals(self, original_stack: np.array):
        for i in range(original_stack.shape[0]):
            cv2.imwrite(self.params.original_dir + self.params.video_name + f"_{self.batch}_{i}.png", original_stack[i])
        print("##### FINISHED WRITING ORIGINALS")
        files_prefix = self.params.original_dir + self.params.video_name + f"_{self.batch}"
        num_frames_in_dir = len(glob.glob(files_prefix))
        print(f"number of saved frames = {num_frames_in_dir}")

    def analyze_ROI(self, ransac_results: RansacResults):
        roi = (ransac_results.inliers[-1][0], ransac_results.inliers[-1][1])
        print(f"ROI is {roi}")

    def is_suspicious(self, ransac_results: RansacResults):
        dec0 = len(np.unique(ransac_results.inliers[:, 0])) > (self.params.num_frames * self.params.pnts_in_trj_limit)
        b_abs = np.abs(np.array([ransac_results.slopeX, ransac_results.slopeY, ransac_results.slopeZ]))
        dec1 = (b_abs[0] ** 2 + b_abs[1] ** 2) ** .5 < self.params.max_trajectory_velocity
        dec2 = (np.max(np.diff(ransac_results.inliers[:, 0])) < self.params.max_t_sus_disappear)
        print(dec0)
        print(dec1)
        print(dec2)
        return dec0 and dec1 and dec2


    #### The Actual Pipeline: ###
    def run_pipeline_dudy(self, original_stack: np.array, params):
        print(f"PIPELINE BATCH {self.batch}")
        # Python version of lambda:  start_index = self.batch == 0 ? 1 : 0 , for branchless execution of eliminating first BGS frame
        start_index = int(self.batch == 0)
        # First BGS frame of first batch will be entirely white, since there is no prior it is all foreground
        start_pipeline = time.time()
        Res_dir_seq = params['ResDir']

        ### Assign new name just for now: ### #TODO: change later
        Mov = original_stack

        ### Stretch & Change Movie For Proper Viewing & Analysis: ###
        Mov = PreProcess_Movie(Mov)

        ### Align Movie: ###
        Mov = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(Mov)

        ### Estimate and substract background: ###
        BG_estimation, Movie_BGS, Movie_BGS_std = BG_estimation_and_substraction_on_aligned_movie(Mov)

        ### Estimate Noise From BG substracted movie (noise estimate per pixel over time): ### #TODO: right now this estimates one noise dc and std for the entire image! return estimation per pixel
        noise_std, noise_dc = NoiseEst(Movie_BGS[:500], params)

        ### use quantile filter to get pixels which are sufficiently above noise for more then a certain quantile of slots in time: ###
        events_points_TXY_indices = Get_Outliers_Above_BG_1(Movie_BGS, Movie_BGS_std, params)

        ### Find trajectories using RANSAC and certain heuristics: ###
        res_points, NonLinePoints, direction_vec, holding_point, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found = \
            Find_Valid_Trajectories_RANSAC_And_Align(events_points_TXY_indices, Movie_BGS, params)

        ### Test whether the trajectory frequency content agrees with the heuristics and is a drone candidate: ###
        DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
            Frequency_Analysis_And_Detection(TrjMov, noise_std, noise_dc, params)

        ### Plot 3D CLoud and Outline Trajectories: ###
        Plot_3D_PointCloud_With_Trajectories(res_points, NonLinePoints, xyz_line, num_of_trj_found, params,
                                             "Plt3DPntCloudTrj", Res_dir_seq)

        ### Plot The Detection Over different frequency bands: ###
        Plot_FFT_Bins_Detection_SubPlots(num_of_trj_found, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec,
                                         params, Res_dir_seq, TrjMov, DetectionDec, DetectionConfLvl)

        ### Show Raw and BG_Substracted_Normalized Videos with proper bounding boxes: ###
        BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Mov, fps=50, tit='Mov', Res_dir=Res_dir_seq,
                                                             flag_save_movie=1, trajectory_tuple=(t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y))
        Plot_BoundingBoxes_On_Video(Mov, fps=50, tit="Mov",
                                    flag_transpose_movie=False,
                                    frame_skip_steps=1, resize_factor=2, flag_save_movie=1,
                                    Res_dir=Res_dir_seq,
                                    histogram_limits_to_stretch=0.01,
                                    trajectory_tuple=(t_vec, trajectory_smoothed_polynom_X,
                                                      trajectory_smoothed_polynom_Y))
        Plot_BoundingBoxes_On_Video(Movie_BGS / Movie_BGS_std, fps=50, tit="Movie_BGS",
                                    flag_transpose_movie=False, frame_skip_steps=1, resize_factor=2,
                                    flag_save_movie=1,
                                    Res_dir=Res_dir_seq,
                                    histogram_limits_to_stretch=0.01,
                                    trajectory_tuple=(
                                    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y))

        ### Save Original and BGS Movie if wanted: ###
        if self.batch == 0:
            if self.params.save_all:
                self.save_originals(Mov)
                self.save_bgs(Movie_BGS)
            self.batch = 1
            return


        end_pipeline = time.time()
        print(f"TOTAL_TIME: {end_pipeline - start_pipeline}\n\n\n\n\n")
        self.batch += 1
