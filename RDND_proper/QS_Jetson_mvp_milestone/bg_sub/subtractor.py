import vpi
import sys
import traceback
import numpy as np


class Subtractor:
    def __init__(self, manager, input_width, input_height, vpi_dtype, learn_rate, threshold, shadow, n_sigma,
                 use_custom_bgs):
        # Program Manager
        self.manager = manager

        # General properties for BG_Sub
        self.width = input_width
        self.height = input_height
        self.vpi_dtype = vpi_dtype

        # Specific algorithm parameters
        self.learn_rate = learn_rate
        self.threshold = threshold
        self.shadow = shadow
        self.n_sigma = n_sigma

        # Initialize VPI Backend
        self.use_custom_bgs = use_custom_bgs

        # TODO: This should happen at the process itself?
        self.__bg_sub = None
        # if self.use_custom_bgs:
        #     with vpi.Backend.CUDA:
        #         self.__bg_sub = vpi.BackgroundSubtractor((self.width, self.height), self.vpi_dtype)

    def bgs_main(self):
        """
        Main method for Background Subtraction subprocess
        Executes an infinite loop of:
            1) Get input data
            2) Execute Background Subtraction on the input
            3) Push the output to Ransac
        """
        if self.use_custom_bgs:
            # TODO: with vpi.Backend.CUDA on the entire loop?
            with vpi.Backend.CUDA:
                self.__bg_sub = vpi.BackgroundSubtractor((self.width, self.height), self.vpi_dtype)

        while True:
            try:
                # Get input frames for BGS
                print("BGS Process waiting for input frames!")
                input_frames = self.manager.get_data_for_bgs()
                print("BGS Process got frames!")

                # Execute BGS from VPI or custom algorithm
                bg_frames = self.__execute_bgs(input_frames)

                # Push the data to ransac process
                print("BGS process pushing data to ransac!")
                # TODO: Time this
                self.manager.push_data_to_ransac(bg_frames)

            except KeyboardInterrupt:
                print("BG Subtractor caught a keyboard interrupt!")
                exit(0)
            except Exception as e:
                print("BG Subtractor caught an unknown exception!")
                print(e)

    # Public methods to debug the Subtractor class
    def debug_bgs(self, input_frames):
        return self.__execute_bgs(input_frames)

    # Private methods to execute the BGS algorithm
    def __custom_bgs(self, Movie_w):
        mean1 = np.mean(Movie_w, 0)
        Input_Vid_BGS = Movie_w - mean1
        var1 = np.mean(Input_Vid_BGS ** 2, 0) ** .5
        bgs_frames = (np.abs(Input_Vid_BGS) - (self.n_sigma * var1))
        return bgs_frames

    def __vpi_bgs(self, arr: np.array):
        try:
            processed_stack = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))
        except MemoryError as e:
            print(f"MEMORY EXCEPTION IS {e}")
            print(f"{traceback.format_exc()}")
            sys.exit(1)

        # with vpi.Backend.CUDA:
        for i in range(arr.shape[0]):
            current_bgs, _ = self.__bg_sub(vpi.asimage(arr[i], self.vpi_dtype), learnrate=self.learn_rate,
                                           threshold=self.threshold, shadow=self.shadow)
            # TODO: Move this out of the loop to save time?
            with current_bgs.rlock():
                processed_stack[i] = current_bgs.cpu()
        return processed_stack

    def __execute_bgs(self, input_frames):
        # Select VPI or custom algorithm
        if self.use_custom_bgs:
            result = self.__custom_bgs(input_frames)
        else:
            result = self.__vpi_bgs(input_frames)
        return result



