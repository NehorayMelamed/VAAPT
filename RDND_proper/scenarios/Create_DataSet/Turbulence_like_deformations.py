from RapidBase.import_all import *



class get_turbulence_flow_field_object:
    def __init__(self,H,W, batch_size, Cn2=2e-13):
        ### Parameters: ###
        h = 100
        # Cn2 = 2e-13
        # Cn2 = 7e-17
        wvl = 5.3e-7
        IFOV = 4e-7
        R = 1000
        VarTiltX = 3.34e-6
        VarTiltY = 3.21e-6
        k = 2 * np.pi / wvl
        r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
        PixSize = IFOV * R
        PatchSize = 2 * r0 / PixSize

        ### Get Current Image Shape And Appropriate Meshgrid: ###
        PatchNumRow = int(np.round(H / PatchSize))
        PatchNumCol = int(np.round(W / PatchSize))
        # shape = I.shape
        [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
        # if I.dtype == 'uint8':
        #     mv = 255
        # else:
        #     mv = 1

        ### Get Random Motion Field: ###
        [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
        [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
        X_large = torch.Tensor(X_large).unsqueeze(-1)
        Y_large = torch.Tensor(Y_large).unsqueeze(-1)
        X_large = (X_large-W/2) / (W/2-1)
        Y_large = (Y_large-H/2) / (H/2-1)

        new_grid = torch.cat([X_large,Y_large],2)
        new_grid = torch.Tensor([new_grid.numpy()]*batch_size)

        self.new_grid = new_grid;
        self.batch_size = batch_size
        self.PatchNumRow = PatchNumRow
        self.PatchNumCol = PatchNumCol
        self.VarTiltX = VarTiltX
        self.VarTiltY = VarTiltY
        self.R = R
        self.PixSize = PixSize
        self.H = H
        self.W = W

    def get_flow_field(self):
        ### TODO: fix this because for Cn2 which is low enough we get self.PatchNumRow & self.PatchNumCol = 0 and this breaks down.....
        ShiftMatX0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltX * self.R / self.PixSize)
        ShiftMatY0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltY * self.R / self.PixSize)
        ShiftMatX0 = ShiftMatX0 * self.W
        ShiftMatY0 = ShiftMatY0 * self.H

        ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')
        ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')

        ShiftMatX = ShiftMatX.squeeze()
        ShiftMatY = ShiftMatY.squeeze()

        return ShiftMatX, ShiftMatY




def turbulence_deformation_single_image_numpy(I, Cn2=5e-13, flag_clip=False):
    ### TODO: why the .copy() ??? ###
    # I = I.copy()

    # I = read_image_default()
    # I = crop_tensor(I,150,150)
    # imshow(I)

    ### Parameters: ###
    h = 100
    # Cn2 = 2e-13
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(I.shape[0] / PatchSize))
    PatchNumCol = int(np.round(I.shape[1] / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

    ### Resize (small) Random Motion Field To Image Size: ###
    ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    ### Add Rescaled Flow Field To Meshgrid: ###
    X = X0 + ShiftMatX
    Y = Y0 + ShiftMatY

    ### Resample According To Motion Field: ###
    I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)

    ### Clip Result: ###
    if flag_clip:
        I = np.minimum(I, mv)
        I = np.maximum(I, 0)

    # imshow(I)

    return I


def turbulence_deformation_single_image_torch(I, flag_clip=False):
    #TODO: make sure this works
    # I = I.copy()

    ### Parameters: ###
    h = 100
    Cn2 = 7e-14
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(I.shape[0] / PatchSize))
    PatchNumCol = int(np.round(I.shape[1] / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

    ### Resize (small) Random Motion Field To Image Size: ###
    ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    ### Add Rescaled Flow Field To Meshgrid: ###
    X = X0 + ShiftMatX
    Y = Y0 + ShiftMatY

    ### Resample According To Motion Field: ###
    I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)

    # # TODO: pytorch implementation of resampling - probably more effective:
    # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='reflection')

    ### Clip Result: ###
    if flag_clip:
        I = np.minimum(I, mv)
        I = np.maximum(I, 0)

    return I




def warp_tensors_affine(input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
    B,C,H,W = input_tensors.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = float32(X0)
    Y0 = float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example;
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)

    #(2). Shift then Scale:
    ### Shift: ###
    shift_x = np.array(shift_x)
    shift_y = np.array(shift_y)
    shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
    shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
    X0 += shift_x_tensor * 1
    Y0 += shift_y_tensor * 1
    ### Scale: ###
    scale = np.array(scale)
    scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
    X0 *= 1 / scale_tensor
    Y0 *= 1 / scale_tensor


    ### Rotation: ###
    X0_max = get_max_correct_form(X0,0).squeeze(2)
    X0_min = get_min_correct_form(X0,0).squeeze(2)
    Y0_max = get_max_correct_form(Y0,0).squeeze(2)
    Y0_min = get_min_correct_form(Y0,0).squeeze(2)
    X0_centered = X0 - (X0_max-X0_min)/2
    Y0_centered = Y0 - (Y0_max-Y0_min)/2
    rotation_angle = np.array(rotation_angle)
    rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
    X0_new = cos(rotation_angle_tensor * pi / 180) * X0_centered - sin(rotation_angle_tensor * pi / 180) * Y0_centered
    Y0_new = sin(rotation_angle_tensor * pi / 180) * X0_centered + cos(rotation_angle_tensor * pi / 180) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear')

    if return_flow_grid:
        return output_tensor, flow_grid
    else:
        return output_tensor






def warp_tensors_affine_inverse(input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
    shift_x = np.array(shift_x)
    shift_y = np.array(shift_y)
    scale = np.array(scale)
    rotation_angle = np.array(rotation_angle)

    shift_x = -shift_x;
    shift_y = -shift_y;
    scale = scale
    rotation_angle = -rotation_angle

    B, C, H, W = input_tensors.shape
    # (1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = float32(X0)
    Y0 = float32(Y0)
    # (2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    # (3). Duplicate meshgrid for each batch Example;
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)


    ### Rotation: ###
    X0_max = get_max_correct_form(X0, 0).squeeze(2)
    X0_min = get_min_correct_form(X0, 0).squeeze(2)
    Y0_max = get_max_correct_form(Y0, 0).squeeze(2)
    Y0_min = get_min_correct_form(Y0, 0).squeeze(2)
    X0_centered = X0 - (X0_max - X0_min) / 2
    Y0_centered = Y0 - (Y0_max - Y0_min) / 2
    rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
    X0_new = cos(rotation_angle_tensor * pi / 180) * X0_centered - sin(rotation_angle_tensor * pi / 180) * Y0_centered
    Y0_new = sin(rotation_angle_tensor * pi / 180) * X0_centered + cos(rotation_angle_tensor * pi / 180) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new


    # (2). Scale then Shift:
    ### Scale: ###
    scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
    X0 *= scale_tensor
    Y0 *= scale_tensor
    ### Shift: ###
    shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
    shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
    X0 += shift_x_tensor * 1
    Y0 += shift_y_tensor * 1


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear')

    if return_flow_grid:
        return output_tensor, flow_grid
    else:
        return output_tensor






def param2theta(param, w, h):
    # h = 50
    # w = 50
    # param = cv2.getRotationMatrix2D((h,w),45,1)

    param = np.linalg.inv(param)
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta


def warp_tensor_affine_matrix(input_tensor, shift_x, shift_y, scale, rotation_angle):
    B,C,H,W = input_tensor.shape
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    center = (width / 2, height / 2)
    affine_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    affine_matrix[0, 2] -= shift_x
    affine_matrix[1, 2] -= shift_y

    ### To Theta: ###
    full_T = np.zeros((3,3));
    full_T[2,2] = 1
    full_T[0:2,:] = affine_matrix;
    theta = param2theta(full_T,W,H)

    affine_matrix_tensor = torch.Tensor(theta)
    affine_matrix_tensor = affine_matrix_tensor.unsqueeze(0) #TODO: multiply batch dimension to batch_size. also and more importantly - generalize to shift_x and affine parameters being of batch_size (or maybe even [batch_size, number_of_channels]) and apply to each tensor and channel

    ### Get Grid From Affine Matrix: ###
    output_grid = torch.nn.functional.affine_grid(affine_matrix_tensor, torch.Size((B,C,H,W)))

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensor, output_grid, mode='bilinear')
    return output_tensor




def warp_numpy_affine(input_mat, shift_x, shift_y, scale, rotation_angle):
    ### Pure OpenCV: ###
    # # shift_scale_rotate(img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    # height, width = input_mat.shape[:2]
    # center = (width / 2, height / 2)
    # affine_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    # affine_matrix[0, 2] += shift_x * width
    # affine_matrix[1, 2] += shift_y * height
    # output_mat = cv2.warpAffine(input_mat, affine_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101, borderValue=None)


    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    # Affine:
    imgaug_parameters['flag_affine_transform'] = True
    imgaug_parameters['affine_scale'] = scale
    imgaug_parameters['affine_translation_percent'] = None
    imgaug_parameters['affine_translation_number_of_pixels'] = {"x":shift_x, "y":shift_y}
    imgaug_parameters['affine_rotation_degrees'] = rotation_angle
    imgaug_parameters['affine_shear_degrees'] = 0
    imgaug_parameters['affine_order'] = cv2.INTER_LINEAR
    imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
    imgaug_parameters['probability_of_affine_transform'] = 1
    ### Get Augmenter: ###
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
    deterministic_Augmenter_Object = imgaug_transforms.to_deterministic()
    output_mat = deterministic_Augmenter_Object.augment_image(input_mat);

    return output_mat

