import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import torch
import cv2
import matplotlib.pylab as pylab
from matplotlib.pylab import figure, colorbar, title
import os
import time

from matplotlib.widgets import Button, Slider


def imshow_torch(image, flag_colorbar=True, title_str=''):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times
    title(title_str)
    return fig

def torch_to_numpy(input_tensor):
    if type(input_tensor) == torch.Tensor:
        (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
        input_tensor = input_tensor.cpu().data.numpy()
        if shape_len == 2:
            #[H,W]
            return input_tensor
        elif shape_len == 3:
            #[C,H,W] -> [H,W,C]
            return np.transpose(input_tensor, [1,2,0])
        elif shape_len == 4:
            #[T,C,H,W] -> [T,H,W,C]
            return np.transpose(input_tensor, [0,2,3,1])
        elif shape_len == 5:
            #[B,T,C,H,W] -> [B,T,H,W,C]
            return np.transpose(input_tensor, [0,1,3,4,2])
    return input_tensor

def get_full_shape_torch(input_tensor):
    if len(input_tensor.shape) == 1:
        W = input_tensor.shape
        H = 1
        C = 1
        T = 1
        B = 1
        shape_len = 1
        shape_vec = (W)
    elif len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
        C = 1
        T = 1
        B = 1
        shape_len = 2
        shape_vec = (H,W)
    elif len(input_tensor.shape) == 3:
        C, H, W = input_tensor.shape
        T = 1
        B = 1
        shape_len = 3
        shape_vec = (C,H,W)
    elif len(input_tensor.shape) == 4:
        T, C, H, W = input_tensor.shape
        B = 1
        shape_len = 4
        shape_vec = (T,C,H,W)
    elif len(input_tensor.shape) == 5:
        B, T, C, H, W = input_tensor.shape
        shape_len = 5
        shape_vec = (B,T,C,H,W)
    shape_vec = np.array(shape_vec)
    return (B,T,C,H,W), shape_len, shape_vec

def imshow_torch_video(input_tensor, number_of_frames=None, FPS=3, flag_BGR2RGB=True, frame_stride=1,
                       flag_colorbar=False, video_title='', video_title_list=None,
                       close_after_next_imshow=False):
    # TODO: fix colorbar
    def get_correct_form(input_tensor, i):
        if shape_len == 4:
            # (T,C,H,W)
            if flag_BGR2RGB and C == 3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 3:
            # (T,H,W)
            output_tensor = torch_to_numpy(input_tensor[i])
        elif shape_len == 5:
            # (B,T,C   ,H,W)
            if flag_BGR2RGB and C == 3:
                output_tensor = cv2.cvtColor(torch_to_numpy(input_tensor[0, i]), cv2.COLOR_BGR2RGB)
            else:
                output_tensor = torch_to_numpy(input_tensor[0, i])
        return output_tensor

    ### Get Parameters: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if number_of_frames is not None:
        number_of_frames_to_show = min(number_of_frames, T)
    else:
        number_of_frames_to_show = T

    # Create figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Initialize variables for current frame and slider position
    current_frame = 0
    current_position = 0

    # Define function for updating the plot
    def update_plot(val):
        nonlocal current_frame
        current_frame = int(val)
        output_tensor = get_correct_form(input_tensor, current_frame)
        im.set_array(output_tensor)
        plt.title(video_title + '  ' + str(current_frame))
        fig.canvas.draw_idle()

    # Define function for previous button click
    def previous_button_clicked(event):
        nonlocal current_frame
        current_frame -= 1
        if current_frame < 0:
            current_frame = 0
        slider.set_val(current_frame)

    # Define function for next button click
    def next_button_clicked(event):
        nonlocal current_frame
        current_frame += 1
        if current_frame > number_of_frames_to_show - 1:
            current_frame = number_of_frames_to_show - 1
        slider.set_val(current_frame)

    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, number_of_frames_to_show - 1, valinit=current_frame, valstep=1)
    slider.on_changed(update_plot)

    # Create previous and next buttons
    ax_prev = plt.axes([0.1, 0.1, 0.05, 0.05])
    ax_next = plt.axes([0.9, 0.1, 0.05, 0.05])
    button_prev = Button(ax_prev, '<')
    button_next = Button(ax_next, '>')
    button_prev.on_clicked(previous_button_clicked)
    button_next.on_clicked(next_button_clicked)

    # Display first frame
    output_tensor = get_correct_form(input_tensor, current_frame)
    im = plt.imshow(output_tensor)

    # Show plot
input_path = 'C:/Users/temp/a.pt'
text_path = 'C:/Users/temp/b.txt'
temp_path = 'C:/Users/temp/temp.txt'

def imshow_torch_seamless():

    img = torch.load(input_path)
    with open(text_path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    flag_colorbar = (lines[0]=='True')
    close_after_next_imshow = (lines[2]=='True')
    imshow_torch(img, flag_colorbar, lines[1])
    plt.show(block=False)
    mtime = os.path.getmtime(temp_path)
    while 1:
        # print(2)
        plt.pause(1)  # <-------
        new_mtime = os.path.getmtime(temp_path)
        if new_mtime - mtime > 0:
            if close_after_next_imshow:
                plt.close()
            return

def imshow_torch_video_seamless():

    video = torch.load(input_path)
    with open(text_path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    number_of_frames = int(lines[0])
    FPS = int(lines[1])
    flag_BGR2RGB = (lines[2] == True)
    frame_stride = int(lines[3])
    close_after_next_imshow = (lines[4] == 'True')
    imshow_torch_video(video, number_of_frames, FPS, flag_BGR2RGB, frame_stride,
                       close_after_next_imshow = close_after_next_imshow)


if __name__ == '__main__':
    mtime = 0

    while 1:
        new_mtime = os.path.getmtime(temp_path)
        if new_mtime - mtime > 0:
            with open(text_path) as f:
                lines = f.readlines()
                lines = [line.rstrip() for line in lines]
            mtime = new_mtime
            if lines[-1] == 'image':
                imshow_torch_seamless()
            else:
                imshow_torch_video_seamless()

