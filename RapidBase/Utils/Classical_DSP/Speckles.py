import torch

from RapidBase.import_all import *


def create_speckles_sequence(final_shape = (100,1,500,500),
                             speckle_size_in_pixels=10,
                             polarization=0,
                             speckle_contrast=0.8,
                             max_decorrelation=0.02,
                             max_beam_wonder=0.05,
                             max_tilt_wonder=0.05):
    # ### TODO: delete, for testing: ###
    # input_tensor = read_image_default_torch()
    # input_tensor = crop_torch_batch(input_tensor, (500, 500))
    # output_tensor = read_video_default_torch()
    # output_tensor = RGB2BW(output_tensor)
    # final_shape = output_tensor.shape
    # # final_shape = (100,1,500,500)
    # T, C, H, W = final_shape
    # speckle_size_in_pixels = 50
    # polarization = 0.
    # max_decorrelation = 0.03
    # max_beam_wonder = 0.0
    # max_tilt_wonder = 0.01  #TODO: maybe present in pixels?
    # speckle_contrast = 1

    surface_phase_factor = 1

    ### Get Parameters: ###
    T, C, H, W = final_shape
    N = H

    # Calculations:
    wf = (N / speckle_size_in_pixels)

    # Create 2D frequency space of size NxN
    x = torch.arange(-N / 2, N / 2, 1)
    [X, Y] = torch.meshgrid(x, x)
    # Assign random values to the frequencies
    beam = torch.exp(- ((X / 2) ** 2 + (Y / 2) ** 2) / wf ** 2)
    # beam = beam / torch.sqrt(sum(sum(abs(beam) ** 2)))
    beam_initial = beam / torch.sqrt((beam.abs() ** 2).sum())

    # Polarization:
    # # Get Surfaces:
    # surface_1 = torch.exp(2 * torch.pi * 1j * 1 * torch.randn(1, C, H, W))
    # surface_2 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    # surface_3 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    # surface_4 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    # Get Surface Phase:
    surface_1_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_2_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_3_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_4_phase = (2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
    surface_1 = torch.exp(surface_1_phase)
    surface_2 = torch.exp(surface_2_phase)
    surface_3 = torch.exp(surface_3_phase)
    surface_4 = torch.exp(surface_4_phase)
    surface_1 = surface_1 - surface_1.mean()
    surface_2 = surface_2 - surface_2.mean()
    surface_3 = surface_3 - surface_3.mean()
    surface_4 = surface_4 - surface_4.mean()
    decorrelation_percent = 0

    ### Initialize Things For Tilt Phase: ###
    # Get tilt phases k-space:
    x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
    y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
    delta_f1 = 1 / W
    delta_f2 = 1 / H
    f_x = x * delta_f1
    f_y = y * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_x = np.fft.fftshift(f_x)
    f_y = np.fft.fftshift(f_y)
    # Build k-space meshgrid:
    [kx, ky] = np.meshgrid(f_x, f_y)
    # get displacement matrix:
    kx = torch.tensor(kx).unsqueeze(0).unsqueeze(0)
    ky = torch.tensor(ky).unsqueeze(0).unsqueeze(0)

    ### Initialize Output Speckles List: ###
    output_frames_list = []
    output_frames_torch = torch.zeros(T, C, H, W)

    ### Get Initial Speckles: ###
    # Get randomly two beams representing wave's polarization
    beam_one = beam_initial * surface_1
    beam_two = beam_initial * surface_2
    # Calculate speckle pattern (practically fft)
    speckle_pattern1 = (torch.fft.fft2(torch.fft.fftshift(beam_one)))
    speckle_pattern2 = (torch.fft.fft2(torch.fft.fftshift(beam_two)))
    # Calculate weighted average of the two beams
    speckle_pattern_total_intensity = (1 - polarization) * speckle_pattern1.abs() ** 2 + polarization * speckle_pattern2.abs() ** 2
    output_frames_torch[0:1, :, :, :] = speckle_pattern_total_intensity
    # output_tensor[0:1, :, :, :] = output_tensor[0:1] * ((1 - speckle_contrast) + speckle_pattern_total_intensity / surface_phase_factor * speckle_contrast)

    ### Get Beam-Wonder & Beam-Tilt Numbers: ###
    # TODO: maybe use get random noise with frequency content
    # TODO: maybe use cumsum twice to get more low frequencies?
    beam_wonder_in_pixels_H = get_random_number_in_range(-max_beam_wonder, max_beam_wonder, array_size=T)
    beam_wonder_in_pixels_W = get_random_number_in_range(-max_beam_wonder, max_beam_wonder, array_size=T)
    beam_wonder_in_pixels_H = np.cumsum(beam_wonder_in_pixels_H)
    beam_wonder_in_pixels_W = np.cumsum(beam_wonder_in_pixels_W)
    # plot(beam_wonder_in_pixels_H); plt.show()
    # plot(beam_wonder_in_pixels_W); plt.show()
    beam_tilt_in_pixels_H = get_random_number_in_range(-max_tilt_wonder, max_tilt_wonder, array_size=T)
    beam_tilt_in_pixels_W = get_random_number_in_range(-max_tilt_wonder, max_tilt_wonder, array_size=T)
    beam_tilt_in_pixels_H = np.cumsum(beam_tilt_in_pixels_H)
    beam_tilt_in_pixels_W = np.cumsum(beam_tilt_in_pixels_W)
    beam_tilt_in_pixels_H = np.cumsum(beam_tilt_in_pixels_H)
    beam_tilt_in_pixels_W = np.cumsum(beam_tilt_in_pixels_W)
    # plot(beam_tilt_in_pixels_H); plt.show()
    # plot(beam_tilt_in_pixels_W); plt.show()
    decorrelation_percent = get_random_number_in_range(0, max_decorrelation, T)
    decorrelation_percent = np.cumsum(decorrelation_percent)
    # plot(decorrelation_percent); plt.show()

    ### Get Surface Phase: ###
    surface_1_phase_list = []
    surface_2_phase_list = []
    surface_1_phase_list.append(surface_1_phase)
    surface_2_phase_list.append(surface_2_phase)
    surface_1_phase_list.append(surface_3_phase)
    surface_2_phase_list.append(surface_4_phase)
    max_number_of_total_decorrelations = np.ceil(decorrelation_percent.max())
    for i in np.arange(1, max_number_of_total_decorrelations):
        surface_1_phase_new = (2 * torch.pi * 1j * 1 * torch.randn(1, C, H, W))
        surface_2_phase_new = (2 * torch.pi * 1j * 1 * torch.randn(1, C, H, W))
        surface_1_phase_list.append(surface_1_phase_new)
        surface_2_phase_list.append(surface_2_phase_new)

    ### Loop Over Frames: ###
    decorrelation_counter = 0
    for frame_index in np.arange(1, T):
        # print(frame_index)
        ### Change The Surface To Create Boiling/Decorrelation: ###
        decorrelation_percent_current = decorrelation_percent[frame_index]
        if decorrelation_percent_current > 1:
            print(str(frame_index) + ', DECORRELATION: ' + str(decorrelation_percent_current))
            # ### Switch Surfaces: ###
            # surface_1 = 0 + copy.deepcopy(surface_3)
            # surface_2 = 0 + copy.deepcopy(surface_4)
            # surface_3 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
            # surface_4 = torch.exp(2 * torch.pi * 1j * 1 * torch.rand(1, C, H, W))
            # ### Switch Surface Phases: ###
            # surface_1_phase = copy.deepcopy(surface_3_phase)
            # surface_2_phase = copy.deepcopy(surface_4_phase)
            # surface_3_phase = (2 * torch.pi * 1j * 10 * torch.rand(1, C, H, W))
            # surface_4_phase = (2 * torch.pi * 1j * 10 * torch.rand(1, C, H, W))

            ### Switch Surface Phases Using Lists: ###
            decorrelation_counter += 1
            decorrelation_percent = decorrelation_percent - 1
            decorrelation_percent_current = decorrelation_percent_current - 1

            # surface_1_phase = surface_1_phase_list[decorrelation_counter]
            # surface_2_phase = surface_2_phase_list[decorrelation_counter]
            # surface_3_phase = surface_1_phase_list[decorrelation_counter + 1]
            # surface_4_phase = surface_2_phase_list[decorrelation_counter + 1]
            # surface_1_total_phase = surface_1_phase * (1 - decorrelation_percent_current) + surface_3_phase * (decorrelation_percent_current)
            # surface_2_total_phase = surface_2_phase * (1 - decorrelation_percent_current) + surface_4_phase * (decorrelation_percent_current)

            # imshow_torch((surface_1_total_phase_previous - surface_1_total_phase).abs())
            # imshow_torch((surface_1_total_phase_previous).abs())
            # (surface_1_phase - surface_1_phase_previous).abs().max()
            # (surface_3_phase - surface_1_phase_previous).abs().max()
            # (surface_3_phase - surface_3_phase_previous).abs().max()
            # (surface_3_phase_previous - surface_1_phase).abs().max()
            # (surface_4_phase_previous - surface_2_phase).abs().max()


        # print(decorrelation_percent_current)
        # ### Add Surfaces: ###
        # surface_1_total = surface_1 * (1 - decorrelation_percent_current) + surface_3 * (decorrelation_percent_current)
        # surface_2_total = surface_2 * (1 - decorrelation_percent_current) + surface_4 * (decorrelation_percent_current)
        ### Add Phases: ###
        surface_1_phase = surface_1_phase_list[decorrelation_counter]
        surface_2_phase = surface_2_phase_list[decorrelation_counter]
        surface_3_phase = surface_1_phase_list[decorrelation_counter + 1]
        surface_4_phase = surface_2_phase_list[decorrelation_counter + 1]
        surface_1_total_phase = surface_1_phase * (1 - decorrelation_percent_current) + surface_3_phase * (decorrelation_percent_current)
        surface_2_total_phase = surface_2_phase * (1 - decorrelation_percent_current) + surface_4_phase * (decorrelation_percent_current)
        # surface_1_total_phase = surface_1_total_phase - surface_1_total_phase.mean()
        # surface_2_total_phase = surface_2_total_phase - surface_1_total_phase.mean()
        surface_1_total = torch.exp(surface_1_total_phase)
        surface_2_total = torch.exp(surface_2_total_phase)
        surface_1_total = surface_1_total - surface_1_total.mean()
        surface_2_total = surface_2_total - surface_2_total.mean()

        ### Change Beam Position (Beam Wonder): ###
        beam_wonder_in_pixels_H_current = wf * beam_wonder_in_pixels_H[frame_index]
        beam_wonder_in_pixels_W_current = wf * beam_wonder_in_pixels_W[frame_index]
        beam = shift_matrix_subpixel_torch(beam_initial, torch.tensor(beam_wonder_in_pixels_W_current), torch.tensor(beam_wonder_in_pixels_H_current))

        ### Change Beam Tilt: ###
        shift_W_tilt = W * beam_tilt_in_pixels_H[frame_index]
        shift_H_tilt = H * beam_tilt_in_pixels_W[frame_index]
        tilt_phase = torch.exp(-(1j * 2 * torch.pi * ky * shift_H_tilt + 1j * 2 * torch.pi * kx * shift_W_tilt))

        ### Get New Speckles: ###
        # Get randomly two beams representing wave's polarization
        beam_one = beam * surface_1_total * tilt_phase
        beam_two = beam * surface_2_total * tilt_phase
        # Calculate speckle pattern (practically fft)
        # speckle_pattern1 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(beam_one)))
        # speckle_pattern2 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(beam_two)))
        speckle_pattern1 = (torch.fft.fft2(torch.fft.fftshift(beam_one)))
        speckle_pattern2 = (torch.fft.fft2(torch.fft.fftshift(beam_two)))
        # Calculate weighted average of the two beams
        speckle_pattern_total_intensity = (1 - polarization) * speckle_pattern1.abs() ** 2 + polarization * speckle_pattern2.abs() ** 2
        output_frames_torch[frame_index:frame_index + 1, :, :, :] = speckle_pattern_total_intensity

    ### Correct for speckle contrast: ###
    output_frames_torch = (1 - speckle_contrast) + output_frames_torch * speckle_contrast

    # imshow_torch_video(output_frames_torch, FPS=25, frame_stride=1)

    return output_frames_torch

# create_speckles_sequence()


def get_speckled_illumination(input_tensor=None, speckle_size_in_pixels=10, polarization=0, max_decorrelation=0.05, max_beam_wonder=0.05, max_tilt_wonder=0.05):
    # ### TODO: delete, for testing: ###
    input_tensor = read_image_default_torch()
    input_tensor = crop_torch_batch(input_tensor, (500, 500))
    input_tensor = read_video_default_torch()
    input_tensor = RGB2BW(input_tensor)
    final_shape = input_tensor.shape
    # final_shape = (100,1,500,500)
    T,C,H,W = final_shape
    speckle_size_in_pixels = 10
    polarization = 0.5
    max_decorrelation = 0.02
    max_beam_wonder = 0.02
    max_tilt_wonder = 0.01
    speckle_contrast = 0.35

    output_speckle_sequence = create_speckles_sequence(final_shape,
                                                     speckle_size_in_pixels,
                                                     polarization,
                                                     speckle_contrast,
                                                     max_decorrelation,
                                                     max_beam_wonder,
                                                     max_tilt_wonder)
    output_tensor = input_tensor * output_speckle_sequence/output_speckle_sequence.max()

    # imshow_torch_video(output_frames_torch, FPS=5, frame_stride=1)
    # imshow_torch_video(BW2RGB(output_tensor*255).clamp(0,255).type(torch.uint8), FPS=5, frame_stride=1)

    return output_tensor

# get_speckled_illumination()
get_speckled_illumination()


