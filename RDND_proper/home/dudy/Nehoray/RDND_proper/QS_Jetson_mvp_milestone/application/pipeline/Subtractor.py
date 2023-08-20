import vpi, sys, traceback
import numpy as np

from application.pipeline.AlgorithmParams import AlgorithmParams


class Subtractor:
    def __init__(self, params: AlgorithmParams):
        with vpi.Backend.CUDA:
            self.bgsub = vpi.BackgroundSubtractor((params.width, params.height), params.vpi_dtype)
        self.params = params
        self.m_format = params.vpi_dtype
        self.learn_rate = params.b_learn_rate
        self.threshold = params.b_threshold
        self.shadow = params.b_shadow

    def custom_bgs(self, Movie_w):
        mean1 = np.mean(Movie_w, 0)
        Input_Vid_BGS = Movie_w - mean1
        var1 = np.mean(Input_Vid_BGS ** 2, 0) ** .5
        bgs_frames = (np.abs(Input_Vid_BGS) - (self.params.b_n_.sigma * var1))
        return bgs_frames

    def vpi_bgs(self, arr: np.array):
        try:
            processed_stack = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))
        except MemoryError as e:
            print(f"MEMORY EXCEPTION IS {e}")
            print(f"{traceback.format_exc()}")
            sys.exit(1)

        # with vpi.Backend.CUDA:
        for i in range(arr.shape[0]):
            current_bgs, _ = self.bgsub(vpi.asimage(arr[i], self.m_format), learnrate=self.learn_rate, threshold=self.threshold,
                                        shadow=self.shadow)
            with current_bgs.rlock():
                processed_stack[i] = current_bgs.cpu()
        return processed_stack

    def perform_bgs(self, arr: np.array):
        #arr = original_arr.copy()
        if self.params.b_use_custom:
            return self.custom_bgs(arr)
        else:
            return self.vpi_bgs(arr)
