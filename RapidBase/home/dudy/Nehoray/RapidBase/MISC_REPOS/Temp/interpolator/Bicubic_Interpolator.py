




### Use Example: ###
#(1). Get images batch:
input_image = read_image_stack_default_torch()
B,C,H,W = input_image.shape
input_image = input_image.cuda()
#(2). Get (delta_x,delta_y) for each batch example:
#   (2.1). define delta_x.shape=[B,H,W] (the same warping for all channels):
delta_x = torch.zeros(input_image.shape[0],input_image.shape[2],input_image.shape[3])
delta_y = torch.zeros(input_image.shape[0],input_image.shape[2],input_image.shape[3])
#   (2.2). define delta_x.shape=[B,C,H,W] (the function implicitely assumes the same warping for all channels...so in order to have a different warp for each channel we will later on make the shape [B,C,H,W]->[B*C,H,W]:
delta_x = torch.zeros(input_image.shape[0], input_image.shape[1], input_image.shape[2],input_image.shape[3])
delta_y = torch.zeros(input_image.shape[0], input_image.shape[1], input_image.shape[2],input_image.shape[3])
delta_x = delta_x.to(input_image.device)
delta_y = delta_y.to(input_image.device)

### Add global shift (differential shift later on): ###
delta_x += -7.8 #[pixels]
delta_y += 5.5 #[pixels]


### Use nn.Module Object to warp images: ###
bicubic_interpolator_layer = Bicubic_Interpolator()
tic()
warped_image_using_layer = bicubic_interpolator_layer(input_image, delta_x, delta_y)
toc()


### Warp Images: ###
tic()
warped_images_bicubic = warp_image(input_image, delta_x, delta_y)
toc()


full_grid = delta_map_to_full_grid_for_bilinear(input_image, delta_x, delta_y)
tic()
warped_images_bilinear = torch.nn.functional.grid_sample(input_image,full_grid, 'bilinear')
toc()

figure(1)
imshow_torch(warped_images_bicubic[0])
figure(2)
imshow_torch(warped_images_bilinear[0])






### To avoid the biggest time consumer - the meshgrid construction - and to easily, explicitely and visibly add this to networks i create a nn.Module object: ###
class Bicubic_Interpolator(nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Bicubic_Interpolator, self).__init__()
        self.X = None
        self.Y = None


    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, delta_x, delta_y):
        # delta_x = map of x deltas from meshgrid, shape=[B,H,W] or [B,C,H,W].... same for delta_Y
        # TODO: later on make this compatible with the needed flexibilities & conventions of deformable convolution
        B, C, H, W = input_image.shape;
        BXC = B * C;

        ### ReOrder delta_x, delta_y: ###
        if len(delta_x.size()) == 3:  # [B,H,W] - > [BXC,H,W,1]
            delta_x = torch.cat([delta_x] * C, dim=0).unsqueeze(-1)
            delta_y = torch.cat([delta_y] * C, dim=0).unsqueeze(-1)
        elif len(delta_x.size()) == 4:  # [B,C,H,W] - > [BXC,H,W,1]
            delta_x = delta_x.view(B * C, H, W).unsqueeze(-1)
            delta_y = delta_y.view(B * C, H, W).unsqueeze(-1)

        ### Create "baseline" meshgrid (as the function ultimately accepts a full map of locations and not just delta's): ###
        #(*). ultimately X.shape=[BXC, H, W, 1]... so check if the input shape has changed and only then create a new meshgrid:
        if self.X is None or self.X.shape[0]!=input_image.shape[0]*input_image.shape[1] or self.X.shape[1]!=input_image.shape[2] or self.X.shape[2]!=input_image.shape[3]:
            [X, Y] = np.meshgrid(np.arange(W), np.arange(H))
            X = torch.Tensor([X] * BXC).unsqueeze(-1)
            Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
            X = X.to(input_image.device)
            Y = Y.to(input_image.device)
            self.X = X;
            self.Y = Y;

        ### Add Difference (delta) Maps to Meshgrid: ###
        # X += delta_x
        # Y += delta_y
        # X = (X - W / 2) / (W / 2 - 1)
        # Y = (Y - H / 2) / (H / 2 - 1)
        new_X = ((self.X+delta_x) - W / 2) / (W / 2 - 1)
        new_Y = ((self.Y+delta_y) - H / 2) / (H / 2 - 1)
        warped_image = bicubic_interpolate(input_image, new_X, new_Y)
        return warped_image




def warp_image(input_image, delta_x, delta_y):
    #[input_image] = [B,C,H,W]
    #[delta_x] = [B,H,W]
    interp_mode = 'bicubic'

    assert len(input_image.size()) == 4 #[B,C,H,W]

    B,C,H,W = input_image.shape;
    BXC = B*C;

    ### ReOrder delta_x, delta_y: ###
    if len(delta_x.size()) == 3: #[B,H,W]
        delta_x = torch.cat([delta_x] * C, dim=0)
        delta_y = torch.cat([delta_y] * C, dim=0)
    elif len(delta_x.size()) == 4: #[B,C,H,W]
        delta_x = delta_x.view(B * C, H, W)
        delta_y = delta_y.view(B * C, H, W)


    ### Add meshgrid "baseline": ###
    # tic()
    [X,Y] = np.meshgrid(np.arange(W), np.arange(H))
    X = torch.Tensor([X]*BXC).unsqueeze(-1)
    Y = torch.Tensor([Y]*BXC).unsqueeze(-1)
    X = X.to(input_image.device)
    Y = Y.to(input_image.device)
    delta_x = delta_x.unsqueeze(-1)
    delta_y = delta_y.unsqueeze(-1)
    X += delta_x
    Y += delta_y
    X = (X - W / 2) / (W / 2 - 1)
    Y = (Y - H / 2) / (H / 2 - 1)
    new_grid = torch.cat([X,Y], 3)
    # toc()

    ### Try Bicubic On Batch: ###
    tic()
    # trans_image = bicubic_interpolate(input_image, new_grid)
    trans_image = bicubic_interpolate(input_image, X, Y)
    toc()

    return trans_image


def bicubic_interpolate(input_image, X, Y):
    ### From [B,C,H,W] -> [B*C,H,W]
    x_shape = input_image.shape;
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
    delta_x_flat = X.contiguous().view(BXC, H*W, 1)
    delta_y_flat = Y.contiguous().view(BXC, H*W, 1)

    ### Flatten completely: [B,H*W,1] -> [B*H*W]: ###
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
    base = torch.zeros(dim1*BXC).int().to(input_image.device) #TODO: changed to dim1*B
    for i in arange(BXC):
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

    output = output.clamp(zero, 1.0)
    output = output.contiguous().reshape(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))

    return output





def delta_map_to_full_grid_for_bilinear(input_image, delta_x, delta_y):
    if len(delta_x.size()) == 4:
        delta_x = delta_x[:,0,:,:]
        delta_y = delta_y[:,0,:,:]

    B, C, H, W = input_image.shape;
    BXC = B * C;
    ### Add meshgrid "baseline": ###
    [X, Y] = np.meshgrid(np.arange(W), np.arange(H))
    X = torch.Tensor([X] * B).unsqueeze(-1)
    Y = torch.Tensor([Y] * B).unsqueeze(-1)
    X = X.to(input_image.device)
    Y = Y.to(input_image.device)
    delta_x = delta_x.unsqueeze(-1)
    delta_y = delta_y.unsqueeze(-1)
    X += delta_x
    Y += delta_y
    X = (X - W / 2) / (W / 2 - 1)
    Y = (Y - H / 2) / (H / 2 - 1)
    new_grid = torch.cat([X, Y], 3)
    return new_grid;





