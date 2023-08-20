import torch
from Cython.Compiler.Nodes import CVarDefNode

from RapidBase.import_all import *


def ImageLoaderCV(path):
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) == 3:  # opencv opens images as  BGR so we need to convert it to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.atleast_3d(image)  # MAKE SURE TO ALWAYS RETURN [H,W,C] FOR CONSISTENCY
    return np.float32(image)


def create_speckles_of_certain_size_in_pixels(final_shape,
                                              speckle_size_in_pixels=10,
                                              polarization=0):
    """
    This function simulates speckles for a gaussian of frequencies, to crate a random 'bloby' pattern
    :param final_shape: Shape of wanted speckles' tensor: Assumes [B,T,C,H,W]. Assumes H = W!!!!!
    :param speckle_size_in_pixels: size of single speckle (determines level of 'variance' between neighboring pixels)
    :param polarization: value in range [0, 1], determines the relative weights of the two polarized vectors
    :return: Tensor of shape final_shape with speckle pattern
    """
    N = final_shape[-1]  # also equals final_shape[-2]

    # Calculations:
    wf = (N / speckle_size_in_pixels)

    # Create 2D frequency space of size NxN
    x = np.arange(-N / 2, N / 2, 1)
    [X, Y] = np.meshgrid(x, x)
    # Assign random values to the frequencies
    beam = np.exp(- ((X / 2) ** 2 + (Y / 2) ** 2) / wf ** 2)
    beam = beam / np.sqrt(sum(sum(abs(beam) ** 2)))

    # Polarization:
    # Get randomly two beams representing wave's polarization
    beam_one = beam * np.exp(2 * np.pi * 1j * 10 * np.random.randn(*final_shape))
    beam_two = beam * np.exp(2 * np.pi * 1j * 10 * np.random.randn(*final_shape))
    # Calculate speckle pattern (practically fft)
    speckle_pattern1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam_one)))
    speckle_pattern2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam_two)))
    # Calculate weighted average of the two beams
    speckle_pattern_total_intensity = (1 - polarization) * abs(speckle_pattern1) * 2 + polarization * abs(
        speckle_pattern2) * 2

    return speckle_pattern_total_intensity


def get_blob_noise_profile(final_shape,
                           blob_size_in_pixels=10,
                           sigma=None,
                           flag_same_on_all_channels=True,
                           flag_same_on_all_batch=True,
                           flag_same_on_all_time=True):
    """
    Get noise profile of a given shape in blob pattern
    :param final_shape: final noise profile shape
    :param blob_size_in_pixels: size of a single 'blob' - determines the level of variance between neighboring pixels
    :param sigma: std of noise pattern
    :param flag_same_on_all_channels: determines whether noise pattern is fixed over channels
    :param flag_same_on_all_batch:determines whether noise pattern is fixed over batch
    :param flag_same_on_all_time:determines whether noise pattern is fixed over time
    :return: blob noise profile of shap final_shape
    """
    # Calculate size of the speckles' tensor - stack non-fixed dimensions
    B, T, C, H, W = final_shape
    speckles_shape = [1, 1, 1, H, W]
    if not flag_same_on_all_batch:
        speckles_shape[0] = B
    if not flag_same_on_all_time:
        speckles_shape[1] = T
    if not flag_same_on_all_channels:
        speckles_shape[2] = C
    # Make square of size max(H, W) x max(H, W) and not rectangle HxW to simplify calculation.
    # Final rectangle will be sliced from the square
    if H > W:
        speckles_shape[4] = speckles_shape[3]
    else:
        speckles_shape[3] = speckles_shape[4]
    # Finally, make size tuple
    speckles_shape = tuple(speckles_shape)

    # Calculate the 'blob' noise pattern using speckles simulation algorithm
    noise_profile = create_speckles_of_certain_size_in_pixels(final_shape=speckles_shape,
                                                              speckle_size_in_pixels=blob_size_in_pixels,
                                                              polarization=0)

    # Convert to tensor
    noise_profile = torch.from_numpy(noise_profile)

    # Normalize the noise profile to sigma
    noise_profile = noise_profile * sigma / noise_profile.std()

    # Slice and stack result tensor to final shape
    # Slice HxW tensor:
    noise_profile = noise_profile[:, :, :, :H, :W]
    # Stack What's Needed:
    if flag_same_on_all_batch:
        noise_profile = torch.cat([noise_profile] * B, dim=0)
    if flag_same_on_all_time:
        noise_profile = torch.cat([noise_profile] * T, dim=1)
    if flag_same_on_all_channels:
        noise_profile = torch.cat([noise_profile] * C, dim=2)

    return noise_profile


def get_1D_noise_from_PSD_torch(final_shape,
                                PSD,
                                sigma,
                                flag_N_samples_from_row_or_col='row',
                                flag_same_on_all_channels=True,
                                flag_same_on_all_batch=False,
                                flag_same_on_all_time=False):
    """
    This function calculates a row/col noise function based on a given PSD.

    Assumptions:
        [PSD] = [H/2] or [Cpsd,H/2] ([W/2] or [Cpsd,W/2]
        [final_shape] = [B,T,C,H,W]
        C1=C2 OR C1=1

    Note: PSD assumed to be double-sided since the signal (noise image) is real.
    therefore, the length is only H/2 or W/2.

    :param final_shape: wanted shape of the noise image
    :param PSD: vector of length H/2 or W/2 respectively (important to ensure!) containing squared values of wanted
    frequencies
    :param sigma: sigma (std) of noise image
    :param flag_N_samples_from_row_or_col: determines whether the PSD corresponds rows or columns
    :param flag_same_on_all_channels: determines whether noise pattern is fixed over channels
    :param flag_same_on_all_batch: determines whether noise pattern is fixed over batch
    :param flag_same_on_all_time: determines whether noise pattern is fixed over time
    :return: noise with the requested frequencies shaped as final_shape
    """
    # Get PSD as Tensor:
    if type(PSD) == np.ndarray:
        PSD = torch.Tensor(PSD)

    # Find he PSDs shape and make it [B,T,C,H] / [B,T,C,W]
    if len(PSD.shape) == 1:  # [H] / [W]
        PSD = PSD.unsqueeze(0)  # [1,H] / [1,W]
    if len(PSD.shape) != 2:
        raise ValueError("PSD must be of shape H/W or CxH/CxW")

    # save number f channels
    Cpsd, _ = PSD.shape
    # Make shape [B,T,C,H] / [B,T,C,W]
    PSD = PSD.unsqueeze(0).unsqueeze(0)

    # Get Final Shape:
    B, T, C, H, W = final_shape

    # Multiply / Stack PSD to fill all needed batch indices:
    if not flag_same_on_all_batch:
        PSD = torch.cat([PSD] * B, dim=0)
    if not flag_same_on_all_channels:
        PSD = torch.cat([PSD] * int(C / Cpsd), dim=2)
    if not flag_same_on_all_time:
        PSD = torch.cat([PSD] * T, dim=1)  # [PSD] = [B,T,C,H]

    # Start calculating the noise:
    # Assign Fpos = sqrt(PSD) to get wanted intensity of each frequency on the positive half
    Fpos = (torch.sqrt(PSD) + 1j * 0)
    # Calculate random phase for each frequency
    phases = torch.randn_like(PSD) * 1000
    # assign values to phases: p => e^(ip)
    phases = torch.cos(phases) + 1j * torch.sin(phases)
    # Multiply by intensities: sqrt(PSD) * e^(ip)
    Fpos *= phases
    # Calculate negative half: Xdf[(N-k) mod N] = X`df[k]
    Fneg = torch.conj(torch.flip(Fpos[:, :, :, :(Fpos.shape[3] - (Fpos.shape[3] % 2))], dims=[3]))
    # Calculate full F
    F = torch.cat((Fpos, Fneg), dim=3)

    # Actually IFFT:
    output = torch.fft.ifftn(F, dim=[-1]).real

    # Stretch To STD=1:
    output = output * 1 / output.std()
    # Scale by sigma:
    output = output * sigma

    # Stack What's Needed:
    if flag_same_on_all_batch:
        output = torch.cat([output] * B, dim=0)
    if flag_same_on_all_channels:
        output = torch.cat([output] * int(C / Cpsd), dim=2)
    if flag_same_on_all_time:
        output = torch.cat([output] * T, dim=1)  # [output] = [B,T,C,H]

    # Get Only number of samples wanted:
    if flag_N_samples_from_row_or_col == 'row':
        number_of_samples = H
    else:
        number_of_samples = W

    output = output[:, :, :, 0:number_of_samples]

    return output


def get_row_col_noise_profile(final_shape,
                              flag_row_or_col='row',
                              flag_totally_random=True,
                              PSD=None,
                              noise_sigma=100,
                              flag_same_on_all_channels=True,
                              flag_same_on_all_batch=True,
                              flag_same_on_all_time=True):
    """
    This function adds row / column non uniformity.
    :param final_shape: final noise profile shape
    :param flag_row_or_col: determines row noise or column noise
    :param flag_totally_random: determines whether the noise is random over rows / columns (or distributed by PSD)
    :param PSD: if not random, determines distribution of noise over rows / columns. Vector of squared frequency
    intensities (energies)
    :param noise_sigma: std of noise
    :param flag_same_on_all_channels: determines whether noise pattern is fixed over channels
    :param flag_same_on_all_batch: determines whether noise pattern is fixed over batch
    :param flag_same_on_all_time: determines whether noise pattern is fixed over time
    :return: noise profile of shape final_shape
    """
    # Get Input Shape:
    B, T, C, H, W = final_shape

    # If Different Noise For Each Row (Not Specific PSD):
    if flag_totally_random:
        # Initialize noise profile
        if flag_row_or_col == 'row':
            noise_profile = torch.ones((1, 1, 1, H, 1))
        elif flag_row_or_col == 'col':
            noise_profile = torch.ones((1, 1, 1, 1, W))

        # Build the block that is profiled randomly (if noise is not uniform in some dimension, this block is multiplied
        # to this dimension's size)
        if not flag_same_on_all_channels:
            noise_profile = torch.cat([noise_profile] * C, dim=2)
        if not flag_same_on_all_batch:
            noise_profile = torch.cat([noise_profile] * B, dim=0)
        if not flag_same_on_all_time:
            noise_profile = torch.cat([noise_profile] * T, dim=1)

        # Make the block random
        noise_profile = torch.randn_like(noise_profile)

        # Multiply the block over the uniform dimensions
        if flag_same_on_all_channels:
            noise_profile = torch.cat([noise_profile] * C, dim=2)
        if flag_same_on_all_batch:
            noise_profile = torch.cat([noise_profile] * B, dim=0)
        if flag_same_on_all_time:
            noise_profile = torch.cat([noise_profile] * T, dim=1)

        # Multiply the block over the rows / columns
        if flag_row_or_col == 'row':
            noise_profile = torch.cat([noise_profile] * W, dim=4)
        elif flag_row_or_col == 'col':
            noise_profile = torch.cat([noise_profile] * H, dim=3)

        # Scale to noise sigma
        noise_image = noise_sigma * noise_profile

        return noise_image

    else:  # Row / col noise profile is not trivial, insert a PSD
        noise_profile = get_1D_noise_from_PSD_torch(final_shape, PSD, noise_sigma,
                                                    flag_N_samples_from_row_or_col=flag_row_or_col,
                                                    flag_same_on_all_channels=flag_same_on_all_channels,
                                                    flag_same_on_all_batch=flag_same_on_all_batch,
                                                    flag_same_on_all_time=flag_same_on_all_time)

        # Multiply the block over the rows / columns
        if flag_row_or_col == 'row':
            noise_profile = noise_profile.unsqueeze(4)
            noise_profile = torch.cat([noise_profile] * W, dim=4)
        elif flag_row_or_col == 'col':
            noise_profile = noise_profile.unsqueeze(3)
            noise_profile = torch.cat([noise_profile] * H, dim=3)

        noise_image = noise_profile
        return noise_image


def get_readout_noise_filenames_and_images(IO_dict):
    """
    Reads external noise images from folder
    :param IO_dict: processing parameters
    :return: list of noise images filenames and optionally the noise images themselves
    """
    if IO_dict.flag_noise_images_to_RAM:
        noise_numpy_images_list, noise_image_filenames_list = read_images_and_filenames_from_folder(
            IO_dict.readout_noise_external_image_path,
            flag_recursive=True,
            crop_size=np.inf,
            max_number_of_images=IO_dict.max_number_of_noise_images,
            allowed_extentions=IO_dict.allowed_extentions,
            flag_return_numpy_or_list='list',
            flag_how_to_concat='T',
            crop_style='random',
            flag_to_BW=False,
            string_pattern_to_search=IO_dict.readout_noise_search_pattern)

    else:
        noise_image_filenames_list = get_image_filenames_from_folder(
            path=IO_dict.readout_noise_external_image_path,
            number_of_images=IO_dict.max_number_of_noise_images,
            allowed_extentions=IO_dict.allowed_extentions,
            flag_recursive=IO_dict.flag_recursive,
            string_pattern_to_search=IO_dict.readout_noise_search_pattern)
        noise_numpy_images_list = None

    return noise_image_filenames_list, noise_numpy_images_list


def get_NU_pattern_filenames_and_images(IO_dict):
    """
    Reads external NU pattern images from folder
    :param IO_dict: processing parameters
    :return: list of noise images filenames and optionally the noise images themselves
    """
    # (1). Offset Pattern:
    if IO_dict.flag_noise_images_to_RAM:
        offset_numpy_images_list, offset_image_filenames_list = read_images_and_filenames_from_folder(
            IO_dict.NU_external_image_offset_path,
            flag_recursive=True,
            crop_size=np.inf,
            max_number_of_images=IO_dict.max_number_of_noise_images,
            allowed_extentions=IO_dict.allowed_extentions,
            flag_return_numpy_or_list='list',
            flag_how_to_concat='T',
            crop_style='random',
            flag_to_BW=False,
            string_pattern_to_search=IO_dict.NU_external_image_offset_search_pattern)

    else:
        offset_image_filenames_list = get_image_filenames_from_folder(
            path=IO_dict.NU_external_image_offset_path,
            number_of_images=IO_dict.max_number_of_noise_images,
            allowed_extentions=IO_dict.allowed_extentions,
            flag_recursive=IO_dict.flag_recursive,
            string_pattern_to_search=IO_dict.NU_external_image_offset_search_pattern)
        offset_numpy_images_list = None

    # (2). Gain Pattern:
    if IO_dict.flag_noise_images_to_RAM:
        gain_numpy_images_list, gain_image_filenames_list = read_images_and_filenames_from_folder(
            IO_dict.NU_external_image_gain_path,
            flag_recursive=True,
            crop_size=np.inf,
            max_number_of_images=IO_dict.max_number_of_noise_images,
            allowed_extentions=IO_dict.allowed_extentions,
            flag_return_numpy_or_list='list',
            flag_how_to_concat='T',
            crop_style='random',
            flag_to_BW=False,
            string_pattern_to_search=IO_dict.NU_external_image_gain_search_pattern)

    else:
        gain_image_filenames_list = get_image_filenames_from_folder(
            path=IO_dict.NU_external_image_gain_path,
            number_of_images=IO_dict.max_number_of_noise_images,
            allowed_extentions=IO_dict.allowed_extentions,
            flag_recursive=IO_dict.flag_recursive,
            string_pattern_to_search=IO_dict.NU_external_image_gain_search_pattern)
        gain_numpy_images_list = None

    return offset_image_filenames_list, offset_numpy_images_list, gain_image_filenames_list, gain_numpy_images_list


def get_dead_pixels_map(final_shape, dead_pixels_fraction,
                        flag_same_on_all_channels=True,
                        flag_same_on_all_batch=True,
                        flag_same_on_all_time=True):
    """
    This function returns a mask of 'dead' pixels distributed normally over the image.
    Uses normal distribution statistics instead of directly picking precise number of pixels to be dead
    :param final_shape: mask final shape, expected [B,T,C,H,W]
    :param dead_pixels_fraction: dead pixels / total pixels
    :param flag_same_on_all_channels: whether dead pixels' distribution fixed over channels
    :param flag_same_on_all_batch: whether dead pixels' distribution fixed over batch
    :param flag_same_on_all_time: whether dead pixels' distribution fixed over time
    :return: dead pixels mask of shape final_shape
    """

    B, T, C, H, W = final_shape
    noise_sigma_map = torch.ones((1, 1, 1, H, W))

    # Multiply / Stack noise map to fill all needed batch indices
    if not flag_same_on_all_batch:
        noise_sigma_map = torch.cat([noise_sigma_map] * B, dim=0)
    if not flag_same_on_all_time:
        noise_sigma_map = torch.cat([noise_sigma_map] * T, dim=1)
    if not flag_same_on_all_channels:
        noise_sigma_map = torch.cat([noise_sigma_map] * C, dim=2)

    # Get Random Gaussian Map For Dead Pixels Statistics
    noise_sigma_map = torch.randn_like(noise_sigma_map)
    threshold_value = np.sqrt(2) * torch.erfinv(torch.Tensor([2 * (1 - dead_pixels_fraction) - 1]))
    dead_pixels_logical_mask = (noise_sigma_map > threshold_value)

    # Stack What's Needed
    if flag_same_on_all_batch:
        dead_pixels_logical_mask = torch.cat([dead_pixels_logical_mask] * B, dim=0)
    if flag_same_on_all_time:
        dead_pixels_logical_mask = torch.cat([dead_pixels_logical_mask] * T, dim=1)
    if flag_same_on_all_channels:
        dead_pixels_logical_mask = torch.cat([dead_pixels_logical_mask] * C, dim=2)

    return dead_pixels_logical_mask


def add_shot_noise(images_to_noise, IO_dict):
    """
    Adds Shot noise to image:
        Each pixel containing N photons is added photons of distribution N(0, sqrt(N))
    Assumes image is normalized to PPP
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    input_images_graylevels = images_to_noise

    # Calculate quantum efficiency:
    input_images_photons = input_images_graylevels * IO_dict.QE * IO_dict.Ge  # TODO make sure addition of Ge needed

    # Calculate shot noise in photons:
    input_images_photons = input_images_photons + torch.sqrt(input_images_photons.clamp(1e-6)) * torch.randn_like(
        input_images_photons)

    # convert back to gray levels
    noisy_images_graylevels = input_images_photons * (1 / IO_dict.QE) * (1 / IO_dict.Ge)

    # TODO return also real noise map
    B, T, C, H, W = images_to_noise.shape
    noise_sigma_map = torch.ones((T, C, H, W))

    return noisy_images_graylevels, noise_sigma_map


def add_dark_current_noise(images_to_noise, IO_dict):
    """
    Adds dark current noise to image:
        For a given constant dark current background DCB, each pixel is added DCB + a where a ~ N(0, sqrt(DCB)):
        x -> x + DCB + a
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    dark_noise_electrons = torch.ones_like(images_to_noise) * IO_dict.dark_current_background
    dark_noise_electrons = dark_noise_electrons + np.sqrt(dark_noise_electrons) * np.randn_like(dark_noise_electrons)

    noisy_images = images_to_noise + dark_noise_electrons

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_per_pixel_NU(images_to_noise, IO_dict):
    """
    Adds per pixel non uniformity to image:
        Relates the variance between different pixels in the transformation from electrons to gray levels.
        For each pixel: x -> A + Bx + Cx^2 + ...
        Where A, B, C are distributed normally
        The noise is fixed over time. Each pixel is assigned different coefficients.
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    # Calculate non uniformity noise per pixel with given polynomial coefficients (sigmas):
    # Non uniformity fixed over channels (?) and time
    B, T, C, H, W = images_to_noise.shape
    input_images_electrons = images_to_noise
    for degree, coefficient in enumerate(IO_dict.PRNU_sigmas_polynomial_params):
        current_gain_map = (coefficient * torch.randn((B, 1, 1, H, W)))
        input_images_electrons += (images_to_noise ** degree) * current_gain_map.to(images_to_noise.device)

    # TODO return also noise map
    noise_sigma_map = torch.ones((T, C, H, W))

    return input_images_electrons, noise_sigma_map


def add_row_NU(images_to_noise, IO_dict):
    """
    Adds per row non uniformity to image:
        Relates the variance between different rows in the transformation from electrons to gray levels.
        For each row: X -> A + BX + CX^2 + ...
        The noise is fixed over time. Each row is assigned different coefficients.
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    noisy_images = images_to_noise
    for degree in np.arange(len(IO_dict.RRNU_sigmas_polynomial_params)):
        NU_row_current_degree = get_row_col_noise_profile(images_to_noise.shape,
                                                          flag_row_or_col='row',
                                                          flag_totally_random=IO_dict.flag_row_NU_random,
                                                          PSD=IO_dict.row_NU_PSD,
                                                          noise_sigma=IO_dict.RRNU_sigmas_polynomial_params[degree],
                                                          flag_same_on_all_channels=IO_dict.flag_same_row_NU_on_all_channels,
                                                          flag_same_on_all_batch=IO_dict.flag_same_row_NU_on_all_batch,
                                                          flag_same_on_all_time=True)

        noisy_images += NU_row_current_degree.to(noisy_images.device) * (noisy_images ** degree)
    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_col_NU(images_to_noise, IO_dict):
    """
    Adds per column non uniformity to image:
        Relates the variance between different columns in the transformation from electrons to gray levels.
        For each column: X -> A + BX + CX^2 + ...
        The noise is fixed over time. Each column is assigned different coefficients.
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    noisy_images = images_to_noise
    for degree in np.arange(len(IO_dict.CRNU_sigmas_polynomial_params)):
        NU_col_current_degree = get_row_col_noise_profile(images_to_noise.shape,
                                                          flag_row_or_col='col',
                                                          flag_totally_random=IO_dict.flag_col_NU_random,
                                                          PSD=IO_dict.col_NU_PSD,
                                                          noise_sigma=IO_dict.CRNU_sigmas_polynomial_params[degree],
                                                          flag_same_on_all_channels=IO_dict.flag_same_col_NU_on_all_channels,
                                                          flag_same_on_all_batch=IO_dict.flag_same_col_NU_on_all_batch,
                                                          flag_same_on_all_time=True)

        noisy_images += NU_col_current_degree.to(noisy_images.device) * (noisy_images ** degree)
    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_blob_NU(images_to_noise, IO_dict):
    """
    Adds non uniformity in blob pattern to image:
        Relates the variance between different areas in the image in the transformation from electrons to gray levels.
        For each pixel: X -> A + BX + CX^2 + ...
        Where A, B, C... are distributed n a bloby pattern
        The noise is fixed over time. Each pixel is assigned different coefficients.
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    noisy_images = images_to_noise
    for degree in np.arange(len(IO_dict.BRNU_sigmas_polynomial_params)):
        NU_blob_current_degree = get_blob_noise_profile(final_shape=images_to_noise.shape,
                                                        blob_size_in_pixels=IO_dict.blob_NU_blob_size,
                                                        sigma=IO_dict.BRNU_sigmas_polynomial_params[degree],
                                                        flag_same_on_all_channels=IO_dict.flag_same_blob_NU_on_all_channels,
                                                        flag_same_on_all_batch=IO_dict.flag_same_blob_NU_on_all_batch,
                                                        flag_same_on_all_time=True)

        noisy_images += NU_blob_current_degree.to(noisy_images.device) * (noisy_images ** degree)
    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_NU_external_image(images_to_noise, IO_dict):
    """
    Adds fixed given non uniformity to image:
        Relates the variance between different pixels in the transformation from electrons to gray levels.
        For each image: X -> offset + gain * X
        The noise is fixed over time. Coefficients are given.
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """

    # Get shape:
    B, T, C, H, W = images_to_noise.shape

    # Get all noise images
    offset_image_filenames_list, \
    offset_numpy_images_list, \
    gain_image_filenames_list, \
    gain_numpy_images_list = get_NU_pattern_filenames_and_images(IO_dict)
    # Choose B random noise images
    noise_indices = get_random_number_in_range(0, len(offset_image_filenames_list) - 1, B).astype('int')

    # Load the chosen images and stack them to a tensor of shape [BxT,Cn,Hn,Wn]
    if IO_dict.flag_noise_images_to_RAM:
        offset_images = torch.stack([torch.from_numpy(offset_numpy_images_list[i]) for i in noise_indices], dim=0)
        gain_images = torch.stack([torch.from_numpy(gain_numpy_images_list[i]) for i in noise_indices], dim=0)
    else:
        # (1). Offset
        offset_images_filenames = [offset_image_filenames_list[i] for i in noise_indices]
        offset_images = []
        for offset_image_path in offset_images_filenames:
            offset_image = ImageLoaderCV(offset_image_path)
            offset_image = torch.from_numpy(np.transpose(offset_image, (2, 0, 1)))
            offset_images.append(offset_image)
        offset_images = torch.stack(offset_images, dim=0)
        # (2). Gain:
        gain_images_filenames = [gain_image_filenames_list[i] for i in noise_indices]
        gain_images = []
        for gain_image_path in gain_images_filenames:
            gain_image = ImageLoaderCV(gain_image_path)
            gain_image = torch.from_numpy(np.transpose(gain_image, (2, 0, 1)))
            gain_images.append(gain_image)
        gain_images = torch.stack(gain_images, dim=0)

    # Make offset and gain match the shape of images_to_noise
    _, Co, Ho, Wo = offset_images.shape
    _, Cg, Hg, Wg = gain_images.shape
    if Ho < H or Hg < H or Wo < W or Wg < W:
        raise ValueError("Shape of non uniformity images gain and offset must match or exceed size of image")
    # If image smaller tan offset / gain, slice to the right size
    offset = offset_images[:, :, :H, :W]
    gain = gain_images[:, :, :H, :W]

    # Increase size where needed
    offset = offset.unsqueeze(1)
    offset = torch.cat([offset] * T, dim=1)
    offset = torch.cat([offset] * int(C / Co), dim=2)

    gain = gain.unsqueeze(1)
    gain = torch.cat([gain] * T, dim=1)
    gain = torch.cat([gain] * int(C / Cg), dim=2)

    # Calculate noisy images
    noisy_images = offset + images_to_noise * gain

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_per_pixel_readout_noise(images_to_noise, IO_dict):
    """
    Adds per pixel readout noise to image:
        Relates the variance between different readouts of images.
        For each pixel: x -> x + A.
        The added noise A is of gaussian distribution over all dimensions
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    # Calculate readout noise per pixel with given polynomial coefficients - equivalent to AWGN
    B, T, C, H, W = images_to_noise.shape

    readout_noise = IO_dict.per_pixel_readout_noise_sigma * torch.randn_like(images_to_noise)
    noisy_images = images_to_noise + readout_noise.to(images_to_noise.device)

    # TODO return also noise map
    noise_sigma_map = torch.ones((T, C, H, W))

    return noisy_images, noise_sigma_map


def add_row_readout_noise(images_to_noise, IO_dict):
    """
    Adds per row readout noise to image:
        Relates the variance between different readouts of images.
        For each row: X -> X + A.
        The added noise A is of gaussian distribution over B,T,C,H dimensions (fixed over W)
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    row_readout_noise_profile = get_row_col_noise_profile(images_to_noise,
                                                          flag_row_or_col='row',
                                                          flag_totally_random=IO_dict.flag_row_readout_noise_random,
                                                          PSD=IO_dict.row_readout_noise_PSD,
                                                          noise_sigma=IO_dict.row_readout_noise_sigma,
                                                          flag_same_on_all_channels=IO_dict.flag_same_row_readout_noise_on_all_channels,
                                                          flag_same_on_all_batch=False,
                                                          flag_same_on_all_time=False)

    noisy_images = images_to_noise + row_readout_noise_profile.to(images_to_noise.device)

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_col_readout_noise(images_to_noise, IO_dict):
    """
    Adds per column readout noise to image:
        Relates the variance between different readouts of images.
        For each column: X -> X + A.
        The added noise A is of gaussian distribution over B,T,C,W dimensions (fixed over H)
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    row_readout_noise_profile = get_row_col_noise_profile(images_to_noise,
                                                          flag_row_or_col='col',
                                                          flag_totally_random=IO_dict.flag_col_readout_noise_random,
                                                          PSD=IO_dict.col_readout_noise_PSD,
                                                          noise_sigma=IO_dict.col_readout_noise_sigma,
                                                          flag_same_on_all_channels=IO_dict.flag_same_col_readout_noise_on_all_channels,
                                                          flag_same_on_all_batch=False,
                                                          flag_same_on_all_time=False)

    noisy_images = images_to_noise + row_readout_noise_profile.to(images_to_noise.device)

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_external_readout_noise(images_to_noise, IO_dict):
    """
    Adds external readout noise:
        Adds to each frame randomly selected noise image from a given collection.
        Noise images are of shape [C,H,W] or [1,H,W]
        Noise over C is fixed if noise images are of shape [1,H,W]
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    B, T, C, H, W = images_to_noise.shape

    # Get all noise images
    noise_images_filenames_list, noise_images_list = get_readout_noise_filenames_and_images(IO_dict)
    # Choose B * T random noise images
    noise_indices = get_random_number_in_range(0, len(noise_images_filenames_list) - 1, B * T).astype('int')

    # Load the chosen images and stack them to a tensor of shape [BxT,Cn,Hn,Wn]
    if IO_dict.flag_noise_images_to_RAM:
        noise_images = torch.stack([torch.from_numpy(noise_images_list[i]) for i in noise_indices], dim=0)
    else:
        noise_images_filenames = [noise_images_filenames_list[i] for i in noise_indices]
        noise_images = []
        for noise_image_path in noise_images_filenames:
            noise_image = ImageLoaderCV(noise_image_path)
            noise_image = torch.from_numpy(np.transpose(noise_image, (2, 0, 1)))
            noise_images.append(noise_image)
        noise_images = torch.stack(noise_images, dim=0)

    # Make noise images' shape [B,T,C,H,W]
    BxT, Cn, Hn, Wn = noise_images.shape
    # Take care of H, W
    if Hn < H or Wn < W:
        raise ValueError("Noise images can't be smaller than original images")
    noise_images = noise_images[:, :, :H, :W]
    # Split BxT to different dimensions
    noise_images = torch.reshape(noise_images, (B, T, Cn, H, W))
    # Stack C if needed
    noise_images = torch.cat([noise_images] * int(C / Cn), dim=2)

    # Add noise to images
    noisy_images = images_to_noise + noise_images

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_dead_pixels_noise(images_to_noise, IO_dict):
    """
    Adds dead pixels to image:
        Takes some fraction of the pixels, normally distributed over the image, and makes them dead (usually black)
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    noisy_images = images_to_noise
    dead_pixels_map = get_dead_pixels_map(final_shape=images_to_noise.shape,
                                          dead_pixels_fraction=IO_dict.dead_pixels_fraction,
                                          flag_same_on_all_channels=IO_dict.flag_same_dead_pixels_on_all_channels,
                                          flag_same_on_all_batch=IO_dict.flag_same_dead_pixels_on_all_batch,
                                          flag_same_on_all_time=True)

    noisy_images[dead_pixels_map] = IO_dict.dead_pixels_value

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_white_pixels_noise(images_to_noise, IO_dict):
    """
    Adds white pixels to image:
        Takes some fraction of the pixels, normally distributed over the image, and makes them white (also possible
        different value)
    :param images_to_noise: images to noise. Assumed shape [B,T,C,H,W]
    :param IO_dict: processing parameters
    :return: noised images
    """
    noisy_images = images_to_noise
    white_pixels_map = get_dead_pixels_map(final_shape=images_to_noise.shape,
                                           dead_pixels_fraction=IO_dict.white_pixels_fraction,
                                           flag_same_on_all_channels=IO_dict.flag_same_white_pixels_on_all_channels,
                                           flag_same_on_all_batch=IO_dict.flag_same_white_pixels_on_all_batch,
                                           flag_same_on_all_time=True)

    noisy_images[white_pixels_map] = IO_dict.white_pixels_value

    # TODO make real noise map
    noise_sigma_map = torch.ones(images_to_noise.shape[1:])

    return noisy_images, noise_sigma_map


def add_noise_to_images_full(images_to_noise, noise_dict, flag_check_c_or_t=True):
    """
    This function adds all types of noise to an image according to flags and parameters in the noise_dict
    Supported noise types:
        Shot noise
        Per pixel non uniformity
        row non uniformity
        column non uniformity
        blob non uniformity
        fixed non uniformity image
        Per pixel readout noise
        Row readout noise
        Column readout noise
        Dead pixels
        White pixels
    :param images_to_noise: images to add noise to
    :param noise_dict: all parameters determining the processing
    :param flag_check_c_or_t: determines whether the flag_how_to_concat is valid. if false, assumes T stacking
    :return: noised images, hopefully noise map too soon :)
    """
    original_images = images_to_noise

    # Make images of uniform shape B,T,C,H,W
    if flag_check_c_or_t:
        if noise_dict.flag_how_to_concat == 'C':
            original_images = Cstack_to_Tstack(original_images, number_of_channels=3)
    # Now that the images are in T stack, make sure they have B dimension
    flag_has_b = True
    if len(original_images.shape) == 4:  # has only T,C,H,W
        flag_has_b = False
        original_images = original_images.unsqueeze(0)

    # Add noise to the image
    noisy_images = original_images
    noise_sigma_map_total = torch.zeros(original_images.shape[1:]).to(original_images.device)

    # Add Shot Noise
    if noise_dict.flag_add_shot_noise:
        noisy_images, noise_sigma_map = add_shot_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Dark Current Noise
    if noise_dict.flag_add_dark_current_noise:
        noisy_images, noise_sigma_map = add_dark_current_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2

    # Add Non Uniformity:
    # Add Per Pixel Non Uniformity
    if noise_dict.flag_add_per_pixel_NU:
        noisy_images, noise_sigma_map = add_per_pixel_NU(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Row Non-Uniformity
    if noise_dict.flag_add_row_NU:
        noisy_images, noise_sigma_map = add_row_NU(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Column Non-Uniformity
    if noise_dict.flag_add_col_NU:
        noisy_images, noise_sigma_map = add_col_NU(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Blob Non-Uniformity
    if noise_dict.flag_add_blob_NU:
        noisy_images, noise_sigma_map = add_blob_NU(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Non-Uniformity Image
    if noise_dict.flag_add_external_image_NU:
        noisy_images, noise_sigma_map = add_NU_external_image(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2

    # Add Readout Noise (Gaussian Noise)
    # Add Per Pixel Readout Noise
    if noise_dict.flag_add_per_pixel_readout_noise:
        noisy_images, noise_sigma_map = add_per_pixel_readout_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Row Readout Noise
    if noise_dict.flag_add_row_readout_noise:
        noisy_images, noise_sigma_map = add_row_readout_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add Column Readout Noise
    if noise_dict.flag_add_col_readout_noise:
        noisy_images, noise_sigma_map = add_col_readout_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add External Readout Noise
    if noise_dict.flag_add_external_readout_noise:
        noisy_images, noise_sigma_map = add_external_readout_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2

    # Add Faulty Pixels
    # Add Dead (Black) Pixels
    if noise_dict.flag_add_dead_pixels_noise:
        noisy_images, noise_sigma_map = add_dead_pixels_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2
    # Add White Pixels
    if noise_dict.flag_add_white_pixels_noise:
        noisy_images, noise_sigma_map = add_white_pixels_noise(noisy_images, noise_dict)
        noise_sigma_map_total += noise_sigma_map ** 2

    # Clamp image to range
    noisy_images = noisy_images.clamp(0, 255)

    # Quantize:
    if noise_dict.flag_quantize_images:
        noisy_images = noisy_images.round()

    # Return the images to their original shape
    # First remove batch if wasn't used
    if flag_has_b is False:
        noisy_images = noisy_images.squeeze(0)
        original_images = original_images.squeeze(0)
    # Change to C stack if needed
    if flag_check_c_or_t and (noise_dict.flag_how_to_concat == 'C'):
        noisy_images = Tstack_to_Cstack(noisy_images)
        original_images = Tstack_to_Cstack(original_images)

    # Get Noise Instance Map (assuming additive for now):
    noise_instance_map = noisy_images - original_images

    # Get Noise Sigma Map: ###
    noise_sigma_map_total = torch.sqrt(noise_sigma_map_total)

    # TODO return real noise map
    return noisy_images, noise_instance_map, noise_sigma_map_total


