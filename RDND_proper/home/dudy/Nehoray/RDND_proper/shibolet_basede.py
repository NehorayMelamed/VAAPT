import torch
import numpy as np
from RDND_proper.RapidBase.import_all import *
import matplotlib.animation as animation

# Create a figure and axis object
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib.pyplot as plt
import numpy as np

### Eilon running plot bad function ####
# def plot_running(y_generator):
#     fig, ax = plt.subplots()
#     x_values = np.arange(1024)
#     # ax.set_ylim(-1 * (10 ** -14),1 * (10 ** -14))
#     line, = ax.plot(x_values, np.zeros_like(x_values))
#     plt.show(block=False)
#
#     for i, y in enumerate(y_generator):
#         y = y.squeeze()
#         y = y * (10 **14)
#         for j in range(len(y)):
#             x = np.arange(j + 1)
#             # print(y.detach().numpy())
#             line.set_data(x, y[:j + 1].detach().numpy())
#             ax.relim()
#             ax.autoscale_view()
#             plt.pause(0.00001)






# Nehoray(actually ChatGPT) better running plot function ####
import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_running(y_generator, display_samples=128, pause_duration=0.000001):
    ### Initialize plot: ###
    fig, ax = plt.subplots()
    colors = ['b', 'g']
    plt.show(block=False)
    last_y = [(0, 'b')] * display_samples

    ### Initialize x axis: ###
    x_vec = np.arange(len(last_y[-display_samples:]))

    im = plt.plot(x_vec, x_vec * 0)[0]

    ### Loop over batches in y_generator (actually y_generated): ###
    for batch_index, y_batch in enumerate(y_generator):
        ###
        y_batch = y_batch.squeeze()
        color = colors[batch_index % 2]

        ### Loop over samples in current batch and insert them into graph: ###
        for y in y_batch:
            ### Add incoming sample and pop most previous sample: ###
            tic()
            last_y.append((y.detach().item(), color))
            last_y.pop(0)
            toc('pop and append')

            ### Get current display_samples last samples values+colors: ###
            tic()
            y_values = [y_value for y_value, _ in last_y[-display_samples:]]
            colors_batch = [color for _, color in last_y[-display_samples:]]
            toc('get y_values and color_values')

            ### Clear axes: ###
            tic()
            ax.clear()
            toc('ax.clear()')

            # ### Loop over each sample in the new (appended + popped) array of current y_values and plot them: ###
            # #TODO: is this necessary? why not simply plot everything at once?
            # for i in range(len(y_values) - 1):
            #     ax.plot(x_vec[i:i+1], y_values[i:i+1], color=colors_batch[i])

            ### Plot new array: ###
            tic()
            ax.plot(x_vec, y_values, color='b')
            toc('plot')

            ### Do stuff chatgpt asked for: ###
            tic()
            ax.relim()
            ax.autoscale_view()
            plt.pause(pause_duration)
            toc('do stuff for axes')

    plt.show()


def y_generator():
    ### Get song: ###
    song = scipy.io.loadmat("/home/simteam-j/Downloads/ofra_haza.mat")["input_vec"]
    # sr, song = scipy.io.wavfile.read("/home/simteam-j/Downloads/shibolet_basade.wav")
    # song = song[:, 20000:]
    song = song[:, 20000:] * 5
    song_shape = song.shape
    N_samples, N_channels = song_shape

    ### Get Parameters: ###
    samples_per_frame = 256
    overlap_samples_per_frame = 128
    Fs = 44100 / 2
    F_lowpass = 2000
    F_highpass = 5000
    filter_number_of_samples = overlap_samples_per_frame
    filter_parameter = 8
    # number_of_batches = int(np.floor(N_samples/overlap_samples_per_frame))
    number_of_batches = 80


    ### Take ROI from the video

    ### Run integranlon that


    ### Initialize FFT_OLA layer: ###
    FFT_OLA_layer = FFT_OLA_PerPixel_Layer_Torch(samples_per_frame=samples_per_frame, filter_name='hann',
                                                 filter_type='lowpass', N=filter_number_of_samples,
                                                 Fs=Fs, low_cutoff=F_lowpass, high_cutoff=F_highpass,
                                                 filter_parameter=filter_parameter)
    actual_filter = FFT_OLA_layer.filter

    ### Loop over audio samples and filter: ###
    output_signal = []
    output_signal_no_filter = []
    signal_buffer = torch.zeros(samples_per_frame)
    previous_buffer_samples = torch.zeros(overlap_samples_per_frame)
    song = song.squeeze()
    print(number_of_batches)
    for batch_counter in np.arange(number_of_batches):
        print(batch_counter)
        ### Get current batch: ###
        start_index = batch_counter * overlap_samples_per_frame
        stop_index = (batch_counter + 1) * overlap_samples_per_frame
        current_batch = song[start_index:stop_index].astype(float)
        current_batch = torch.tensor(current_batch)
        output_signal_no_filter.append(current_batch)

        ### Use Buffering ###
        signal_buffer = torch.cat([previous_buffer_samples, current_batch], dim=0)

        ### Get to 5D (the filtering function is general and is able to filter image pixels in time as well, later on a 1D function will be created if needed): ###
        signal_buffer = torch_get_5D(signal_buffer, 'T')

        ### Filter current batch: ###
        filtered_batch = FFT_OLA_layer.forward(signal_buffer)
        yield filtered_batch
        ### Append to output list: ###
        # output_signal.append(filtered_batch)

        ### Assign previous buffer with current output: ###
        # previous_buffer_samples = filtered_batch.squeeze()
        # previous_buffer_samples = current_batch



plot_running(y_generator())


### Concat all samples: ###
# import sounddevice as sd
# final_output_signal = torch.cat(output_signal, dim=1).squeeze()
# final_input_signal = torch.cat(output_signal_no_filter, dim=0).squeeze()
# final_output_signal = final_output_signal[257:]
# final_output_signal_numpy = final_output_signal.detach().cpu().numpy()
# sd.play(final_output_signal_numpy.astype(int16)*20, Fs)
# sd.play(final_input_signal.cpu().numpy().astype(int16), Fs)
# sd.play(song[0:44100*4], Fs)
# plot_torch(final_output_signal[0:10000])
# plot_torch(final_input_signal[257:])
# plt.legend(['filtered signal', 'original signal'])
# plt.figure()
# plot(song[:len(final_output_signal_numpy)])
# plt.show()
#
#
# frame_count = 0
# current_frames = torch.zeros((1, song_shape[1]))
# filtered_song = torch.zeros((1, song_shape[1]))
# buffer = torch.zeros((256, song_shape[1]))
# song = numpy_to_torch(song[13000:]).squeeze()
# torch_filter = torch_get_3D(torch.tensor(filter_dltd.num, dtype=torch.float32), input_dims="H")
# fps = 100
# conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=512, bias=False)
# conv.weight.data = torch_filter
# for frame_count in range(song.shape[0]):
#     frame = song[0]
#     if frame_count % 256 == 0 and frame_count > 0:
#         buffer = torch.cat((buffer, current_frames), dim=0)
#         current_frames = frame.unsqueeze(dim=0)
#         filtered_buffer = conv(buffer[1:-1])
#         # filtered_song = torch.cat((filtered_song, filtered_buffer))
#         yield torch_fft_and_fftshift(filtered_buffer)[0][0].abs()**2
#         buffer = buffer[257:]
#     current_frames = torch.cat((current_frames, frame.unsqueeze(dim=0)), dim=0)

# sr, song = scipy.io.wavfile.read("/home/simteam-j/Downloads/shibolet_basade.wav")
# filter_name = 'hann'
# filter_type = 'low'
# filter_dltd = get_filter_1D(filter_name, filter_type, 512, 512, 0.8, 3)
# song_shape = song.shape
# frame_count = 0
# current_frames = torch.zeros((1, song_shape[1]))
# filtered_song = torch.zeros((1, song_shape[1]))
# buffer = torch.zeros((256, song_shape[1]))
# song = numpy_to_torch(song[13000:]).squeeze()
# torch_filter = torch_get_3D(torch.tensor(filter_dltd.num, dtype=torch.float32), input_dims="H")
# fps = 100
# conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=512, bias=False)
# conv.weight.data = torch_filter
# for frame_count in range(song.shape[0]):
#     frame = song[0]
#     if frame_count % 256 == 0 and frame_count > 0:
#         buffer = torch.cat((buffer, current_frames), dim=0)
#         current_frames = frame.unsqueeze(dim=0)
#         filtered_buffer = conv(buffer[1:-1])
#         # filtered_song = torch.cat((filtered_song, filtered_buffer))
#         print(torch_fft_and_fftshift(filtered_buffer))
#         buffer = buffer[257:]
#     current_frames = torch.cat((current_frames, frame.unsqueeze(dim=0)), dim=0)
#
