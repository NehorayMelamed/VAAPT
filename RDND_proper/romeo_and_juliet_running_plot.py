import matplotlib.pyplot as plt

from RDND_proper.RapidBase.import_all import *

### functions for roi selection ###
def get_roi_from_user(frame):
    fig, ax = plt.subplots(1)
    ax.imshow(frame)

    roi = plt.ginput(2)  # Get two points from the user
    plt.close()

    x1, y1 = roi[0]
    x2, y2 = roi[1]
    roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    return roi

def crop_video_array(video_array, roi):
    # Assuming input video_array shape is (T, H, W, C)
    # and roi is a tuple of (x, y, w, h)
    x, y, w, h = map(int, roi)  # Convert ROI values to integers
    return video_array[:, y:y+h, x:x+w, :]

def crop_video_interactive(video_array):
    # Get the first frame
    first_frame = video_array[0]

    # Get the ROI from the user
    roi = get_roi_from_user(first_frame)

    # Crop the video array using the selected ROI
    cropped_video_array = crop_video_array(video_array, roi)

    return cropped_video_array

def plot_running(y_generator, number_of_samples_to_display=128*4, title="", mini_batch_size=3, number_of_samples_to_pause=10):
    ### Default parameters: ###

    ### Initialize stuff: ###
    x_vec = np.linspace(0, number_of_samples_to_display, num=number_of_samples_to_display)
    colors = ['b', 'g']
    X,Y = np.meshgrid(x_vec,x_vec)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    last_y = [0] * number_of_samples_to_display
    line, = ax.plot([])
    plt.title(title)
    ax.set_xlim(x_vec.min(), x_vec.max())
    ax.set_ylim(-2.5,2.5)
    fig.canvas.draw()   # note that the first draw comes before setting data

    ### cache the background: ###
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
    plt.show(block=False)

    ### Loop over batches in y_generator (in our case only 1 batch so for loop isn't even necessary here): ###
    for batch_index, y_batch in enumerate(y_generator):
        ### get current batch: ###
        y_batch = y_batch.squeeze()

        ### Loop over current batch samples sample-by-sample: ####
        number_of_mini_batches = len(y_batch) // mini_batch_size
        for sample in range(number_of_mini_batches):
            ### Set current line data: ###
            line.set_data(x_vec, last_y)

            ### Pop most previous and append latest sample: ###
            #TODO: understand if there's a more effective way to do this
            del last_y[0:mini_batch_size]
            last_y.extend(list(y_batch[sample*mini_batch_size:(sample+1)*mini_batch_size].detach().numpy()))

            ### restore background: ###
            fig.canvas.restore_region(axbackground)

            ### redraw just the points: ###
            ax.draw_artist(line)

            fig.canvas.blit(ax.bbox)


            if sample % number_of_samples_to_pause == 0:
                plt.pause(0.000001)

        plt.pause(0.000000000001)




### Continue here to shibulet
def y_generator():
    ### Get song: ###
    # load video
    input_tensor = torch.load("/home/mafat/Desktop/eylon/RDND_proper/with_movement_mat.pt")
    input_tensor_as_numpy = torch_to_numpy(input_tensor)

    ## Select ROI: ###
    roi_tensor_numpy = crop_video_interactive(input_tensor_as_numpy)
    roi_tensor_torch = numpy_to_torch(roi_tensor_numpy)
    roi_tensor_total_intensity = roi_tensor_torch.mean([-1, -2]).squeeze()
    roi_tensor_total_intensity_DC_substracted = roi_tensor_total_intensity - roi_tensor_total_intensity.mean(0, True)

    T = roi_tensor_total_intensity.shape[0]

    ### Get Parameters For Filtering: ###
    samples_per_frame = 256
    overlap_samples_per_frame = 128
    Fs = 200
    F_lowpass = 20
    F_highpass = 250
    filter_number_of_samples = overlap_samples_per_frame
    filter_parameter = 8
    number_of_batches = int(np.floor(T/overlap_samples_per_frame))
    # number_of_batches = 80

    ### Initialize FFT_OLA layer: ###
    FFT_OLA_layer = FFT_OLA_PerPixel_Layer_Torch(samples_per_frame=samples_per_frame,
                                                 filter_name='hann',
                                                 filter_type='lowpass',
                                                 N=filter_number_of_samples,
                                                 Fs=Fs,
                                                 low_cutoff=F_lowpass,
                                                 high_cutoff=F_highpass,
                                                 filter_parameter=filter_parameter)

    ### Loop over audio samples and filter: ###
    output_signal_unfiltered = []
    output_signal_filtered = []
    # signal_buffer = torch.zeros(samples_per_frame)
    previous_buffer_samples = torch.zeros(overlap_samples_per_frame)
    # song = song.squeeze()
    print(number_of_batches)

    for batch_counter in np.arange(number_of_batches):
        print(batch_counter)
        ### Get current batch: ###
        start_index = batch_counter * overlap_samples_per_frame
        stop_index = (batch_counter + 1) * overlap_samples_per_frame
        current_batch = roi_tensor_total_intensity_DC_substracted[start_index:stop_index]
        current_batch = torch.tensor(current_batch)

        ### Save unfiltered signal for later: ###
        output_signal_unfiltered.append(current_batch)

        ### Use Buffering ###
        signal_buffer = torch.cat([previous_buffer_samples, current_batch], dim=0)

        ### Get to 5D (the filtering function is general and is able to filter image pixels in time as well, later on a 1D function will be created if needed): ###
        signal_buffer = torch_get_5D(signal_buffer, 'T')

        ### Filter current batch: ###
        filtered_batch = FFT_OLA_layer.forward(signal_buffer)

        ### Save filtered signal for later: ###
        output_signal_filtered.append(filtered_batch)

        ### Update previous batch for next iteration: ###
        previous_buffer_samples = current_batch

        ### Get output filtered signal as output for generator: ###
        yield filtered_batch

    # ### Get complete filtered signal: ###
    # output_signal_filtered = torch.cat(output_signal_filtered, 1).squeeze()
    # output_signal_unfiltered = torch.cat(output_signal_unfiltered)
    # plot_torch(output_signal_filtered)
    # plot_torch(output_signal_unfiltered)


plot_running(y_generator(), title="romeo and juliet")