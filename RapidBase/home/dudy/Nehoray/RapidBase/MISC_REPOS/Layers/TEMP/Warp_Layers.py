from RapidBase.import_all import *


def bicubic_interpolate(input_image, X, Y):
    ### From [B,C,H,W] -> [B*C,H,W]
    x_shape = input_image.shape
    B,C,H,W = input_image.shape
    BXC = x_shape[0]*x_shape[1]
    input_image = input_image.contiguous().reshape(-1, int(x_shape[2]), int(x_shape[3])) #[B,C,H,W]->[B*C,H,W]

    # height = new_grid.shape[1]
    # width = new_grid.shape[2]
    height = input_image.shape[1]
    width = input_image.shape[2]

    ### Reshape & Extract delta maps: ###
    # theta_flat = new_grid.contiguous().view(new_grid.shape[0], height * width, new_grid.shape[3])  #[B,H*W,2] - spatial flattening
    # delta_x_flat = theta_flat[:, :, 0:1]
    # delta_y_flat = theta_flat[:, :, 1:2]
    ###
    if X.shape[0] != input_image.shape[0]: #input_image.shape=[B,C,H,W] but X.shape=[B,H,W,1]  --> X.shape=[BXC,H,W,1]
        X = X.repeat([C,1,1,1])
        Y = Y.repeat([C,1,1,1])
    delta_x_flat = X.contiguous().view(BXC, H*W, 1)
    delta_y_flat = Y.contiguous().view(BXC, H*W, 1)

    ### Flatten completely: [BXC,H*W,1] -> [B*H*W]: ###
    x_map = delta_x_flat.contiguous().view(-1)
    y_map = delta_y_flat.contiguous().view(-1)
    x_map = x_map.float()
    y_map = y_map.float()
    height_f = float(height)
    width_f = float(width)


    ### Take Care of needed symbolic variables: ###
    zero = 0
    max_y = int(height - 1)
    max_x = int(width - 1)
    ###
    x_map = (x_map + 1) * (width_f - 1) / 2.0  #Here i divide again by 2?!!?!....then why multiply by 2 in the first place?!
    y_map = (y_map + 1) * (height_f - 1) / 2.0
    ###
    x0 = x_map.floor().int()
    y0 = y_map.floor().int()
    ###
    xm1 = x0 - 1
    ym1 = y0 - 1
    ###
    x1 = x0 + 1
    y1 = y0 + 1
    ###
    x2 = x0 + 2
    y2 = y0 + 2
    ###
    tx = x_map - x0.float()
    ty = y_map - y0.float()


    ### the coefficients are for a=-1/2
    c_xm1 = ((-tx ** 3 + 2 * tx ** 2 - tx) / 2.0)
    c_x0 = ((3 * tx ** 3 - 5 * tx ** 2 + 2) / 2.0)
    c_x1 = ((-3 * tx ** 3 + 4 * tx ** 2 + tx) / 2.0)
    c_x2 = (1.0 - (c_xm1 + c_x0 + c_x1))

    c_ym1 = ((-ty ** 3 + 2 * ty ** 2 - ty) / 2.0)
    c_y0 = ((3 * ty ** 3 - 5 * ty ** 2 + 2) / 2.0)
    c_y1 = ((-3 * ty ** 3 + 4 * ty ** 2 + ty) / 2.0)
    c_y2 = (1.0 - (c_ym1 + c_y0 + c_y1))

    # TODO: pad image for bicubic interpolation and clamp differently if necessary
    xm1 = xm1.clamp(zero, max_x)
    x0 = x0.clamp(zero, max_x)
    x1 = x1.clamp(zero, max_x)
    x2 = x2.clamp(zero, max_x)

    ym1 = ym1.clamp(zero, max_y)
    y0 = y0.clamp(zero, max_y)
    y1 = y1.clamp(zero, max_y)
    y2 = y2.clamp(zero, max_y)

    dim2 = width
    dim1 = width * height

    ### Take care of indices base for flattened indices: ###
    #TODO: avoid using a for loop
    base = torch.zeros(dim1*BXC).int().to(input_image.device) #TODO: changed to dim1*B
    for i in np.arange(BXC):
        base[(i+1)*H*W : (i+2)*H*W] = torch.Tensor([(i+1)*H*W]).to(input_image.device).int()

    base_ym1 = base + ym1 * dim2
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    base_y2 = base + y2 * dim2

    idx_ym1_xm1 = base_ym1 + xm1
    idx_ym1_x0 = base_ym1 + x0
    idx_ym1_x1 = base_ym1 + x1
    idx_ym1_x2 = base_ym1 + x2

    idx_y0_xm1 = base_y0 + xm1
    idx_y0_x0 = base_y0 + x0
    idx_y0_x1 = base_y0 + x1
    idx_y0_x2 = base_y0 + x2

    idx_y1_xm1 = base_y1 + xm1
    idx_y1_x0 = base_y1 + x0
    idx_y1_x1 = base_y1 + x1
    idx_y1_x2 = base_y1 + x2

    idx_y2_xm1 = base_y2 + xm1
    idx_y2_x0 = base_y2 + x0
    idx_y2_x1 = base_y2 + x1
    idx_y2_x2 = base_y2 + x2

    #(*). TODO: thought: this flattening coupled with torch.index_select assumes that number of elements in input_image_flat = number of elements in the indices (like idx_y2_xm1)...but input_imag_flat includes Channels!!!....
    input_image_flat = input_image.contiguous().view(-1).float()

    I_ym1_xm1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_ym1_xm1.long()))
    I_ym1_x0 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_ym1_x0.long()))
    I_ym1_x1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_ym1_x1.long()))
    I_ym1_x2 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_ym1_x2.long()))

    I_y0_xm1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y0_xm1.long()))
    I_y0_x0 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y0_x0.long()))
    I_y0_x1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y0_x1.long()))
    I_y0_x2 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y0_x2.long()))

    I_y1_xm1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y1_xm1.long()))
    I_y1_x0 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y1_x0.long()))
    I_y1_x1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y1_x1.long()))
    I_y1_x2 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y1_x2.long()))

    I_y2_xm1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y2_xm1.long()))
    I_y2_x0 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y2_x0.long()))
    I_y2_x1 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y2_x1.long()))
    I_y2_x2 = torch.index_select(input_image_flat, dim=0, index=Variable(idx_y2_x2.long()))

    output_ym1 = c_xm1 * I_ym1_xm1 + c_x0 * I_ym1_x0 + c_x1 * I_ym1_x1 + c_x2 * I_ym1_x2
    output_y0 = c_xm1 * I_y0_xm1 + c_x0 * I_y0_x0 + c_x1 * I_y0_x1 + c_x2 * I_y0_x2
    output_y1 = c_xm1 * I_y1_xm1 + c_x0 * I_y1_x0 + c_x1 * I_y1_x1 + c_x2 * I_y1_x2
    output_y2 = c_xm1 * I_y2_xm1 + c_x0 * I_y2_x0 + c_x1 * I_y2_x1 + c_x2 * I_y2_x2

    #TODO: changed from height,width to B,height,width
    output = c_ym1.view(BXC,height, width) * output_ym1.view(BXC,height, width) +\
             c_y0.view(BXC,height, width) * output_y0.view(BXC,height, width) + \
             c_y1.view(BXC,height, width) * output_y1.view(BXC,height, width) + \
             c_y2.view(BXC,height, width) * output_y2.view(BXC,height, width)

    # output = output.clamp(zero, 1.0)  #TODO: why clamp?!?!!?
    output = output.contiguous().reshape(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
    return output






class Warp_Object(nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Warp_Object, self).__init__()
        self.X = None
        self.Y = None

    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear'):
        # delta_x = map of x deltas from meshgrid, shape=[B,H,W] or [B,C,H,W].... same for delta_Y
        B, C, H, W = input_image.shape
        BXC = B * C

        ### ReOrder delta_x, delta_y: ###
        #TODO: this expects delta_x,delta_y to be image sized tensors. but sometimes i just wanna pass in a single number per image
        #(1). Dim=3 <-> [B,H,W], I Interpret As: Same Flow On All Channels:
        if len(delta_x.shape) == 3:  # [B,H,W] - > [B,H,W,1]
            delta_x = delta_x.unsqueeze(-1)
            delta_y = delta_x.unsqueeze(-1)
            flag_same_on_all_channels = True
        #(2). Dim=4 <-> [B,C,H,W], Different Flow For Each Channel:
        elif (len(delta_x.shape) == 4 and delta_x.shape[1]==C):  # [B,C,H,W] - > [BXC,H,W,1] (because pytorch's function only warps all channels of a tensor the same way so in order to warp each channel seperately we need to transfer channels to batch dim)
            delta_x = delta_x.view(B * C, H, W).unsqueeze(-1)
            delta_y = delta_y.view(B * C, H, W).unsqueeze(-1)
            flag_same_on_all_channels = False
        #(3). Dim=4 but C=1 <-> [B,1,H,W], Same Flow On All Channels:
        elif len(delta_x.shape) == 4 and delta_x.shape[1]==1:
            delta_x = delta_x.permute([0,2,3,1]) #[B,1,H,W] -> [B,H,W,1]
            delta_y = delta_y.permute([0,2,3,1])
            flag_same_on_all_channels = True
        #(4). Dim=4 but C=1 <-> [B,H,W,1], Same Flow On All Channels:
        elif len(delta_x.shape) == 4 and delta_x.shape[3] == 1:
            flag_same_on_all_channels = True


        ### Create "baseline" meshgrid (as the function ultimately accepts a full map of locations and not just delta's): ###
        #(*). ultimately X.shape=[BXC, H, W, 1]/[B,H,W,1]... so check if the input shape has changed and only then create a new meshgrid:
        flag_input_changed_from_last_time = (self.X is None) or (self.X.shape[0]!=BXC and flag_same_on_all_channels==False) or (self.X.shape[0]!=B and flag_same_on_all_channels==True) or (self.X.shape[1]!=H) or (self.X.shape[2]!=W)
        if flag_input_changed_from_last_time:
            print('new meshgrid')
            [X, Y] = np.meshgrid(np.arange(W), np.arange(H))  #X.shape=[H,W]
            if flag_same_on_all_channels:
                X = torch.Tensor([X] * B).unsqueeze(-1) #X.shape=[B,H,W,1]
                Y = torch.Tensor([Y] * B).unsqueeze(-1)
            else:
                X = torch.Tensor([X] * BXC).unsqueeze(-1) #X.shape=[BXC,H,W,1]
                Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
            X = X.to(input_image.device)
            Y = Y.to(input_image.device)
            self.X = X
            self.Y = Y


        # [X, Y] = np.meshgrid(np.arange(W), np.arange(H))
        # X = torch.Tensor([X] * BXC).unsqueeze(-1)
        # Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
        # X = X.to(input_image.device)
        # Y = Y.to(input_image.device)


        ### Add Difference (delta) Maps to Meshgrid: ###
        ### Previous Try: ###
        # X += delta_x
        # Y += delta_y
        # X = (X - W / 2) / (W / 2 - 1)
        # Y = (Y - H / 2) / (H / 2 - 1)
        # ### Previous Use: ###
        # new_X = ((self.X + delta_x) - W / 2) / (W / 2 - 1)
        # new_Y = ((self.Y + delta_y) - H / 2) / (H / 2 - 1)
        ### New Use: ###
        new_X = 2 * ((self.X + delta_x)) / max(W-1,1) - 1
        new_Y = 2 * ((self.Y + delta_y)) / max(H-1,1) - 1
        # ### No Internal Tensors: ###
        # new_X = 2 * ((X + delta_x)) / max(W - 1, 1) - 1
        # new_Y = 2 * ((Y + delta_y)) / max(H - 1, 1) - 1

        if flag_bicubic_or_bilinear == 'bicubic':
            #input_image.shape=[B,C,H,W] , new_X.shape=[B,H,W,1] OR new_X.shape=[BXC,H,W,1]
            warped_image = bicubic_interpolate(input_image, new_X, new_Y)
            return warped_image
        else:
            bilinear_grid = torch.cat([new_X,new_Y],dim=3)
            if flag_same_on_all_channels:
                #input_image.shape=[B,C,H,W] , bilinear_grid.shape=[B,H,W,2]
                input_image_to_bilinear = input_image
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                return warped_image
            else:
                #input_image.shape=[BXC,1,H,W] , bilinear_grid.shape=[BXC,H,W,2]
                input_image_to_bilinear = input_image.reshape(-1, int(H), int(W)).unsqueeze(1) #[B,C,H,W]->[B*C,1,H,W]
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                warped_image = warped_image.view(B,C,H,W)
                return warped_image


class Warp_Object_OpticalFlow(Warp_Object):
    # Initialize this with a module
    def __init__(self):
        super(Warp_Object_OpticalFlow, self).__init__()
        self.X = None
        self.Y = None

    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, optical_flow_map, flag_bicubic_or_bilinear='bilinear', flag_input_type='XY'):
        if flag_input_type == 'XY':
            return super().forward(input_image, optical_flow_map[:,0:1,:,:], optical_flow_map[:,1:2,:,:], flag_bicubic_or_bilinear)
        elif flag_input_type == 'HW':
            return super().forward(input_image, optical_flow_map[:, 1:2, :, :], optical_flow_map[:, 0:1, :, :], flag_bicubic_or_bilinear)



# ### Compare github implementation and mine: ###
# ### Use Example: ###
# #(1). Get images batch:
# # input_image = read_image_stack_default_torch()
# input_image = read_image_default_torch()
# input_image = crop_torch_batch(input_image, 1000)
#
# B,C,H,W = input_image.shape
# input_image = input_image.cuda()
# #(2). Get (delta_x,delta_y) for each batch example:
# #   (2.1). define delta_x.shape=[B,H,W] (the same warping for all channels):
# delta_x = torch.zeros(input_image.shape[0],input_image.shape[2],input_image.shape[3])
# delta_y = torch.zeros(input_image.shape[0],input_image.shape[2],input_image.shape[3])
# #   (2.2). define delta_x.shape=[B,C,H,W] (the function implicitely assumes the same warping for all channels...so in order to have a different warp for each channel we will later on make the shape [B,C,H,W]->[B*C,H,W]:
# delta_x = torch.zeros(input_image.shape[0], input_image.shape[1], input_image.shape[2],input_image.shape[3])
# delta_y = torch.zeros(input_image.shape[0], input_image.shape[1], input_image.shape[2],input_image.shape[3])
# delta_x = delta_x.to(input_image.device)
# delta_y = delta_y.to(input_image.device)
# ### Add global shift (differential shift later on): ###
# delta_x += -0.5 #[pixels]
# delta_y += 0.5 #[pixels]
#
# ### My Implementation: ###
# from RapidBase.import_all import *
# warp_interpolator_object_bilinear = Warp_Object()
# warp_interpolator_object_bicubic = Warp_Object()
# # tic()
# warped_image_using_layer_bilinear = warp_interpolator_object_bilinear(input_image, delta_x, delta_y, 'bilinear')
# # toc('bilinaer ')
# # tic()
# warped_image_using_layer_bicubic = warp_interpolator_object_bicubic(input_image, delta_x, delta_y, 'bicubic')
# # toc('bicubic ')
# # ### Github Implementation: ###
# # warped_image_using_function_bilinear = billinear_warp(input_image.view(B*C,1,H,W), -1*torch.cat([delta_x.view(B*C,1,H,W), delta_y.view(B*C,1,H,W)],dim=1))
# # warped_image_using_function_bilinear = warped_image_using_function_bilinear.view(3,3,H,W)
# ### Show Results: ###
# # imshow_torch(warped_image_using_layer_bilinear[0])
# # imshow_torch(warped_image_using_function_bilinear[0])
# # imshow_torch(warped_image_using_layer_bicubic[0])
# # imshow_torch(warped_image_using_layer_bicubic-warped_image_using_layer_bilinear)


















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

        self.new_grid = new_grid
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




def warp_tensor_affine(input_tensor, shift_x, shift_y, scale, rotation_angle):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensor.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)


    # #(1). Scale then Shift:
    # ### Scale: ###
    # X0 *= 1/scale
    # Y0 *= 1/scale
    # ### Shift: ###
    # X0 += shift_x * scale
    # Y0 += shift_y * scale

    #(2). Shift then Scale:
    ### Shift: ###
    X0 += shift_x * 1
    Y0 += shift_y * 1
    ### Scale: ###
    X0 *= 1 / scale
    Y0 *= 1 / scale

    ### Rotation: ###
    # X0_centered = X0 - X0.max() / 2
    # Y0_centered = Y0 - Y0.max() / 2
    # X0_centered = X0 - W / 2
    # Y0_centered = Y0 - H / 2
    X0_centered = X0 - (X0.max()-X0.min())/2
    Y0_centered = Y0 - (Y0.max()-Y0.min())/2
    rotation_angle = torch.Tensor([rotation_angle])
    X0_new = np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered - np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    Y0_new = np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered + np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensor, flow_grid, mode='bilinear')
    return output_tensor








def warp_tensor_affine_inverse(input_tensor, shift_x, shift_y, scale, rotation_angle):
    shift_x = -shift_x
    shift_y = -shift_y
    scale = scale
    rotation_angle = -rotation_angle

    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensor.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)

    ### Rotation: ###
    # X0_centered = X0 - X0.max() / 2
    # Y0_centered = Y0 - Y0.max() / 2
    # X0_centered = X0 - W / 2
    # Y0_centered = Y0 - H / 2
    X0_centered = X0 - (X0.max() - X0.min()) / 2
    Y0_centered = Y0 - (Y0.max() - Y0.min()) / 2
    rotation_angle = torch.Tensor([rotation_angle])
    X0_new = np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered - np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    Y0_new = np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered + np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new

    #(1). Scale then Shift:
    ### Scale: ###
    X0 *= scale
    Y0 *= scale
    ### Shift: ###
    X0 += shift_x * 1
    Y0 += shift_y * 1

    # #(2). Shift then Scale:
    # ### Shift: ###
    # X0 += shift_x * 1
    # Y0 += shift_y * 1
    # ### Scale: ###
    # X0 *= 1 / scale
    # Y0 *= 1 / scale




    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensor, flow_grid, mode='bilinear')
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
    full_T = np.zeros((3,3))
    full_T[2,2] = 1
    full_T[0:2,:] = affine_matrix
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
    # height, width = input_mat.shape[:2]ft_x, shift_y, scale, rotation_angle):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensor.shape
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
    output_mat = deterministic_Augmenter_Object.augment_image(input_mat)

    return output_mat







def vec_to_pytorch_format(input_vec, dim_to_put_scalar_in=0):
    if dim_to_put_scalar_in == 1: #channels
        output_tensor = torch.Tensor(input_vec).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    if dim_to_put_scalar_in == 0: #batches
        output_tensor = torch.Tensor(input_vec).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return output_tensor



def get_max_correct_form(input_tensor, dim_to_leave):
    ### Assuming input_tensor.shape = [B,C,H,W] or 4D Tensor in any case: ###

    if dim_to_leave==0: #get max value for each batch example (max among all values of that batch example including all channels)
        values, indices = input_tensor.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        values = values.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #Return to pytorch format: [B,1,1,1]
    if dim_to_leave==1:  # get max value for each channel (max among all values of including batch dimension)
        values, indices = input_tensor.max(0)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        values = values.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]
    if dim_to_leave==(0,1) or dim_to_leave==[0,1]:  # get max value of each feature map
        if len(values.shape) > 2:
            values, indices = input_tensor.max(2)
        if len(values.shape) > 1:
            values, indices = values.max(2)
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [B,C,1,1]

    return values



def get_min_correct_form(input_tensor, dim_to_leave):
    ### Assuming input_tensor.shape = [B,C,H,W] or 4D Tensor in any case: ###

    if dim_to_leave==0: #get max value for each batch example (max among all values of that batch example including all channels)
        values, indices = input_tensor.min(1)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        values = values.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #Return to pytorch format: [B,1,1,1]
    if dim_to_leave==1:  # get max value for each channel (max among all values of including batch dimension)
        values, indices = input_tensor.min(0)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        values = values.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]
    if dim_to_leave==(0,1) or dim_to_leave==[0,1]:  # get max value of each feature map
        if len(values.shape) > 2:
            values, indices = values.min(2)
        if len(values.shape) > 1:
            values, indices = values.min(2)
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [B,C,1,1]

    return values


import numpy as np
import torch
class Warp_Tensors_Affine_Layer(nn.Module):
    def __init__(self, *args):
        super(Warp_Tensors_Affine_Layer, self).__init__()
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.X0 = None
        self.Y0 = None
    def forward(self, input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False, flag_interpolation_mode='bilinear'):
        flag_new_meshgrid = self.X0 is None
        if self.X0 is not None:
            flag_new_meshgrid = flag_new_meshgrid and (self.X0.shape[-1] != input_tensors.shape[-1] or self.X0.shape[-2] != input_tensors.shape[-2])
        if flag_new_meshgrid:
            self.B, self.C, self.H, self.W = input_tensors.shape
            # (1). Create meshgrid:
            [self.X0, self.Y0] = np.meshgrid(np.arange(self.W), np.arange(self.H))
            self.X0 = np.float32(self.X0)
            self.Y0 = np.float32(self.Y0)
            # (2). Turn meshgrid to be tensors:
            self.X0 = torch.Tensor(self.X0).to(input_tensors.device)
            self.Y0 = torch.Tensor(self.Y0).to(input_tensors.device)
            self.X0 = self.X0.unsqueeze(0)
            self.Y0 = self.Y0.unsqueeze(0)
            # (3). Duplicate meshgrid for each batch Example
            self.X0 = torch.cat([self.X0] * self.B, 0)
            self.Y0 = torch.cat([self.Y0] * self.B, 0)

        ### Make Sure All Inputs Are In The Correct Format: ###
        if type(shift_x) == np.ndarray or type(shift_x) == np.float32 or type(shift_x) == np.float64:
            shift_x = np.atleast_1d(shift_x)
            shift_y = np.atleast_1d(shift_y)
            shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
            shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
        elif type(shift_x) == torch.Tensor:
            shift_x_tensor = torch.atleast_3d(shift_x).to(input_tensors.device)
            shift_y_tensor = torch.atleast_3d(shift_y).to(input_tensors.device)

        if type(rotation_angle) == np.ndarray or type(rotation_angle) == np.float32 or type(rotation_angle) == np.float64:
            rotation_angle = np.atleast_1d(rotation_angle)
            rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
        elif type(rotation_angle) == torch.Tensor:
            rotation_angle_tensor = torch.atleast_3d(rotation_angle).to(input_tensors.device)

        if type(scale) == np.ndarray or type(scale) == np.float32 or type(scale) == np.float64:
            scale = np.atleast_1d(scale)
            scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
        elif type(scale) == torch.Tensor:
            scale_tensor = torch.atleast_3d(scale).to(input_tensors.device)

        # (2). Shift then Scale:
        ### Shift: ###
        X1 = self.X0 + shift_x_tensor * 1
        Y1 = self.Y0 + shift_y_tensor * 1
        ### Scale: ###
        X1 *= 1 / scale_tensor
        Y1 *= 1 / scale_tensor

        ### Rotation: ###
        # X0_max = get_max_correct_form(X1, 0).squeeze(2)
        # X0_min = get_min_correct_form(X1, 0).squeeze(2)
        # Y0_max = get_max_correct_form(Y1, 0).squeeze(2)
        # Y0_min = get_min_correct_form(Y1, 0).squeeze(2)
        X0_max = X1[0,-1,-1]
        X0_min = X1[0,0,0]
        Y0_max = Y1[0, -1, -1]
        Y0_min = Y1[0, 0, 0]
        X0_centered = X1 - (X0_max - X0_min) / 2   #TODO: make sure this is correct, perhapse we need an odd number of elements and rotate around it? don't know
        Y0_centered = Y1 - (Y0_max - Y0_min) / 2
        ### TODO: maybe i can speed things up here?!?!?
        X0_new = torch.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - torch.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
        Y0_new = torch.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + torch.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
        X1 = X0_new
        Y1 = Y0_new

        ### Normalize Meshgrid to 1 to conform with grid_sample: ###
        #TODO: i don't know how long it takes to do unsqueeze(-1)
        X1 = X1.unsqueeze(-1)
        Y1 = Y1.unsqueeze(-1)
        X1 = X1 / ((self.W - 1) / 2)
        Y1 = Y1 / ((self.H - 1) / 2)
        flow_grid = torch.cat([X1, Y1], 3)

        ### Warp: ###
        if flag_interpolation_mode == 'bilinear':
            output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid.to(input_tensors.device), mode='bilinear')
        elif flag_interpolation_mode == 'bicubic':
            output_tensor = bicubic_interpolate(input_tensors, X1, Y1)

        if return_flow_grid:
            return output_tensor, flow_grid
        else:
            return output_tensor


class Warp_Tensors_Inverse_Affine_Layer(nn.Module):
    def __init__(self, *args):
        super(Warp_Tensors_Inverse_Affine_Layer, self).__init__()
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.X0 = None
        self.Y0 = None

    def forward(self, input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
        if self.X0 is None:
            self.B, self.C, self.H, self.W = input_tensors.shape
            # (1). Create meshgrid:
            [self.X0, self.Y0] = np.meshgrid(np.arange(self.W), np.arange(self.H))
            self.X0 = np.float32(self.X0)
            self.Y0 = np.float32(self.Y0)
            # (2). Turn meshgrid to be tensors:
            self.X0 = torch.Tensor(self.X0)
            self.Y0 = torch.Tensor(self.Y0)
            self.X0 = self.X0.unsqueeze(0)
            self.Y0 = self.Y0.unsqueeze(0)
            # (3). Duplicate meshgrid for each batch Example
            self.X0 = torch.cat([self.X0] * self.B, 0)
            self.Y0 = torch.cat([self.Y0] * self.B, 0)

        shift_x = np.array(shift_x)
        shift_y = np.array(shift_y)
        scale = np.array(scale)
        rotation_angle = np.array(rotation_angle)

        shift_x = -shift_x
        shift_y = -shift_y
        scale = scale
        rotation_angle = -rotation_angle

        ### Rotation: ###
        X0_max = get_max_correct_form(self.X0, 0).squeeze(2)
        X0_min = get_min_correct_form(self.X0, 0).squeeze(2)
        Y0_max = get_max_correct_form(self.Y0, 0).squeeze(2)
        Y0_min = get_min_correct_form(self.Y0, 0).squeeze(2)
        X0_centered = self.X0 - (X0_max - X0_min) / 2
        Y0_centered = self.Y0 - (Y0_max - Y0_min) / 2
        rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
        X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
        Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
        X1 = X0_new
        Y1 = Y0_new

        # (2). Scale then Shift:
        ### Scale: ###
        scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
        X1 *= scale_tensor
        Y1 *= scale_tensor
        ### Shift: ###
        shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
        shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
        X1 += shift_x_tensor * 1
        Y1 += shift_y_tensor * 1

        ### Normalize Meshgrid to 1 to conform with grid_sample: ###
        X1 = X1.unsqueeze(-1)
        Y1 = Y1.unsqueeze(-1)
        X1 = X1 / ((self.W - 1) / 2)
        Y1 = Y1 / ((self.H - 1) / 2)
        flow_grid = torch.cat([X1, Y1], 3)

        ### Warp: ###
        output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear')

        if return_flow_grid:
            return output_tensor, flow_grid
        else:
            return output_tensor



def warp_tensors_affine(input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensors.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
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
    X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
    Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
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

    shift_x = -shift_x
    shift_y = -shift_y
    scale = scale
    rotation_angle = -rotation_angle

    B, C, H, W = input_tensors.shape
    # (1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    # (2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    # (3). Duplicate meshgrid for each batch Example
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
    X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
    Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
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





# ### Test Warping on batches of tensors: ###
# shift_x = [0,10,-20]
# shift_y = [-10,30,0]
# scale = [1,2,0.5]
# rotation_angle = [0,-30,30]
# input_tensors = read_image_stack_default_torch()
# output_tensors = warp_tensors_affine(input_tensors, shift_x=shift_x, shift_y=shift_y, scale=scale, rotation_angle=rotation_angle)
# ### Test warp_tensors_affine: ###
# figure(1)
# imshow_torch(input_tensors[0,:,:,:])
# figure(2)
# imshow_torch(output_tensors[0,:,:,:])
# figure(3)
# imshow_torch(input_tensors[1,:,:,:])
# figure(4)
# imshow_torch(output_tensors[1,:,:,:])
# figure(5)
# imshow_torch(input_tensors[2,:,:,:])
# figure(6)
# imshow_torch(output_tensors[2,:,:,:])


# ### Test warp_tensors_affine_inverse: ###
# figure(1)
# imshow_torch(input_tensors[0,:,:,:])
# figure(2)
# imshow_torch(output_tensors_realign[0,:,:,:])
# figure(3)
# imshow_torch(input_tensors[1,:,:,:])
# figure(4)
# imshow_torch(output_tensors_realign[1,:,:,:])
# figure(5)
# imshow_torch(input_tensors[2,:,:,:])
# figure(6)
# imshow_torch(output_tensors_realign[2,:,:,:])
#
#
#
#
#
#
#
#
# #### Affine Transform On GPU: ###
# crop_size = 100
# batch_size = 1
# input_mat = read_image_default()
# input_mat = crop_tensor(input_mat, crop_size,crop_size)
# current_image_transposed = np.transpose(input_mat, [2,0,1])
# input_tensor = torch.Tensor(current_image_transposed).unsqueeze(0)
#
# ### Affine Parameters: ###
# shift_x = 10
# shift_y = 0
# scale = 1
# rotation_angle = 0
#
# ### Test pytorch affine warping function: ###
# # output_tensor = warp_tensor_affine_matrix(input_tensor, shift_x=shift_x, shift_y=shift_y, scale=scale, rotation_angle=rotation_angle)
# output_tensor = warp_tensor_affine(input_tensor, shift_x=shift_x, shift_y=shift_y, scale=scale, rotation_angle=rotation_angle)
# figure(1)
# imshow_torch(input_tensor[0,:,:,:])
# figure(2)
# imshow_torch(output_tensor[0,:,:,:])
#
# ### Compare to OpenCV: ###
# output_mat = warp_numpy_affine(input_mat, shift_x=-shift_x, shift_y=-shift_y, scale=scale, rotation_angle=-rotation_angle)
# figure(3)
# imshow(output_mat)
# colorbar()
#
# figure(4)
# output_mat_transposed = np.transpose(output_mat, [2,0,1])
# output_mat_to_tensor = torch.Tensor(output_mat_transposed).unsqueeze(0)
# imshow_torch(abs(output_mat_to_tensor - output_tensor))
#
#
# ### ReAlign Deformed Mat in pytorch: ###
# output_mat_tensor_realigned = warp_tensor_affine_inverse(output_mat_to_tensor, shift_x=shift_x, shift_y=shift_y, scale=scale, rotation_angle=rotation_angle)
# figure(5)
# imshow_torch(output_mat_tensor_realigned)
# figure(6)
# imshow_torch(abs(output_mat_tensor_realigned-input_tensor))




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
    # # Perspective:
    # imgaug_parameters['flag_perspective_transform'] = False
    # imgaug_parameters['flag_perspective_transform_keep_size'] = True
    # imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
    # imgaug_parameters['probability_of_perspective_trans#Perspective:
    # imgaug_parameters['flag_perspective_transform'] = False
    # imgaug_parameters['flag_perspective_transform_keep_size'] = True
    # imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
    # imgaug_parameters['probability_of_perspective_transform'] = 1form'] = 1
    ### Get Augmenter: ###
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
    deterministic_Augmenter_Object = imgaug_transforms.to_deterministic()
    output_mat = deterministic_Augmenter_Object.augment_image(input_mat);

    return output_mat











