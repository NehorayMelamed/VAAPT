import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *



################################################################################################################################################################################################################################################
#### CSPN: ####
class CPSN_V1(nn.Module):
    def __init__(self, number_of_refinement_iterations, propagation_kernel_size):
        super(CPSN_V1, self).__init__()
        self.number_of_refinement_iterations = number_of_refinement_iterations
        self.propagation_kernel_size = propagation_kernel_size
        self.number_of_input_channels = 1  # depth number of channels is 1...obviously
        self.number_of_output_channels = 1  # we output refined depth so again numebr of channels is 1

    def forward(self, guidance, blur_depth, sparse_depth=None): #input 8 guidance maps in the form of 8 channels (assuming we diffuse an affinity of 3x3 neighborhood)

        # normalize features
        gate1_w1_cmb = torch.abs(guidance.narrow(1, 0, 1))
        gate2_w1_cmb = torch.abs(guidance.narrow(1, 1, 1))
        gate3_w1_cmb = torch.abs(guidance.narrow(1, 2, 1))
        gate4_w1_cmb = torch.abs(guidance.narrow(1, 3, 1))
        gate5_w1_cmb = torch.abs(guidance.narrow(1, 4, 1))
        gate6_w1_cmb = torch.abs(guidance.narrow(1, 5, 1))
        gate7_w1_cmb = torch.abs(guidance.narrow(1, 6, 1))
        gate8_w1_cmb = torch.abs(guidance.narrow(1, 7, 1))

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            result_depth = (1 - sparse_mask) * blur_depth.clone() + sparse_mask * sparse_depth

        for i in range(self.number_of_refinement_iterations):
            # one propagation
            elewise_max_gate1 = self.eight_way_propagation(gate1_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate2 = self.eight_way_propagation(gate2_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate3 = self.eight_way_propagation(gate3_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate4 = self.eight_way_propagation(gate4_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate5 = self.eight_way_propagation(gate5_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate6 = self.eight_way_propagation(gate6_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate7 = self.eight_way_propagation(gate7_w1_cmb, result_depth, self.propagation_kernel_size)
            elewise_max_gate8 = self.eight_way_propagation(gate8_w1_cmb, result_depth, self.propagation_kernel_size)

            result_depth = self.max_of_8_tensor(elewise_max_gate1, elewise_max_gate2, elewise_max_gate3,
                                                elewise_max_gate4, \
                                                elewise_max_gate5, elewise_max_gate6, elewise_max_gate7,
                                                elewise_max_gate8)

            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth.clone() + sparse_mask * sparse_depth

        return result_depth



    def eight_way_propagation(self, weight_matrix, blur_matrix, kernel):
        [batch_size, channels, height, width] = weight_matrix.size()
        self.avg_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel, stride=1, padding=(kernel - 1) // 2, bias=False)
        weight = torch.ones(1, 1, kernel, kernel).cuda()

        self.avg_conv.weight = nn.Parameter(weight)
        for param in self.avg_conv.parameters():
            param.requires_grad = False

        self.sum_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel, stride=1, padding=(kernel - 1) // 2, bias=False)
        sum_weight = torch.ones(1, 1, kernel, kernel).cuda()
        self.sum_conv.weight = nn.Parameter(sum_weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False
        weight_sum = self.sum_conv(weight_matrix)
        avg_sum = self.avg_conv((weight_matrix * blur_matrix))

        out = torch.div(avg_sum, weight_sum)
        return out


    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4)








class CSPN_V2(nn.Module):

    def __init__(self, number_of_refinement_iterations, propagation_kernel_size):
        super(CSPN_V2, self).__init__()
        self.number_of_refinement_iterations = number_of_refinement_iterations
        self.propagation_kernel_size = propagation_kernel_size
        self.number_of_input_channels = 1 #depth number of channels is 1...obviously
        self.number_of_output_channels = 1 #we output refined depth so again numebr of channels is 1

        ### Padding Layers: ###
        self.left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))  # the padding notation is (left,right,top,bottom)  (assuming we diffuse an affinity of 3x3 neighborhood)
        self.center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        self.right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        self.left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        self.right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        self.left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        self.center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        self.right_bottom_pad = nn.ZeroPad2d((2, 0, 2, 0))


    def forward(self, guidance, blurred_depth, sparse_depth=None):  #input 8 guidance maps in the form of 8 channels
        ### Get Absolute Of Guidance Features: ###
        #(*). guidance.narrow explanation: basically: go to dim=1 (channels), start at i*self.number_of_output_channels, get self.number_of_output_channels elements
        #     so basically they used .narrow() instead of guidance[:,i,:,:].... maybe because it's faster?
        gate1_wb_cmb = torch.abs(guidance.narrow(1, 0 * self.number_of_output_channels, self.number_of_output_channels))
        gate2_wb_cmb = torch.abs(guidance.narrow(1, 1 * self.number_of_output_channels, self.number_of_output_channels))
        gate3_wb_cmb = torch.abs(guidance.narrow(1, 2 * self.number_of_output_channels, self.number_of_output_channels))
        gate4_wb_cmb = torch.abs(guidance.narrow(1, 3 * self.number_of_output_channels, self.number_of_output_channels))
        gate5_wb_cmb = torch.abs(guidance.narrow(1, 4 * self.number_of_output_channels, self.number_of_output_channels))
        gate6_wb_cmb = torch.abs(guidance.narrow(1, 5 * self.number_of_output_channels, self.number_of_output_channels))
        gate7_wb_cmb = torch.abs(guidance.narrow(1, 6 * self.number_of_output_channels, self.number_of_output_channels))
        gate8_wb_cmb = torch.abs(guidance.narrow(1, 7 * self.number_of_output_channels, self.number_of_output_channels))

        ### Gates/Maps Affinity To Center Pixel Roles: ###
        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        ### Pad Features: ###
        #(1). top pad
        gate1_wb_cmb = self.left_top_pad(gate1_wb_cmb).unsqueeze(1)
        gate2_wb_cmb = self.center_top_pad(gate2_wb_cmb).unsqueeze(1)
        gate3_wb_cmb = self.right_top_pad(gate3_wb_cmb).unsqueeze(1)
        #(2). center pad
        gate4_wb_cmb = self.left_center_pad(gate4_wb_cmb).unsqueeze(1)
        gate5_wb_cmb = self.right_center_pad(gate5_wb_cmb).unsqueeze(1)
        #(3). bottom pad
        gate6_wb_cmb = self.left_bottom_pad(gate6_wb_cmb).unsqueeze(1)
        gate7_wb_cmb = self.center_bottom_pad(gate7_wb_cmb).unsqueeze(1)
        gate8_wb_cmb = self.right_bottom_pad(gate8_wb_cmb).unsqueeze(1)


        ### Concatenate Padded Guidance Features In Channels Index: ###
        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb,
                             gate5_wb_cmb, gate6_wb_cmb, gate7_wb_cmb, gate8_wb_cmb), 1)


        ### Refine: ####
        raw_depth_input = blurred_depth
        result_depth = blurred_depth
        if sparse_depth is not None:  ### Placeholder for possible future where we get some kind of sparse "GT" depth from our device
            sparse_mask = sparse_depth.sign()
        ### Loop over refinement iterations: ###
        for i in range(self.number_of_refinement_iterations):
            ### Pad Depth: ###
            SPN_kernel_size = self.propagation_kernel_size
            result_depth = self.pad_blurred_depth(result_depth)  #we keep padding inefficiently...simply use padded convolution layer (reflective) or don't do it and crop GT according to valid pixels...what's weird is the Specific Padding locations....
                                                                 #OH....their probably doint it on purpose!!!!...there are 8 neighbors for each pixel and we want a different wait for each neighbor or something like that!!#$
            ### 3D Convolution: ###
            neigbor_weighted_sum = self.eight_way_propagation(gate_wb, result_depth, SPN_kernel_size)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum
            if sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth



    def pad_blurred_depth(self, blurred_depth):
        ### top pad ###
        blurred_depth_1 = self.left_top_pad(blurred_depth).unsqueeze(1)
        blurred_depth_2 = self.center_top_pad(blurred_depth).unsqueeze(1)
        blurred_depth_3 = self.right_top_pad(blurred_depth).unsqueeze(1)
        ### center pad ###
        blurred_depth_4 = self.left_center_pad(blurred_depth).unsqueeze(1)
        blurred_depth_5 = self.right_center_pad(blurred_depth).unsqueeze(1)
        ### bottom pad ###
        blurred_depth_6 = self.left_bottom_pad(blurred_depth).unsqueeze(1)
        blurred_depth_7 = self.center_bottom_pad(blurred_depth).unsqueeze(1)
        blurred_depth_8 = self.right_bottom_pad(blurred_depth).unsqueeze(1)

        result_depth = torch.cat((blurred_depth_1, blurred_depth_2, blurred_depth_3, blurred_depth_4,
                                  blurred_depth_5, blurred_depth_6, blurred_depth_7, blurred_depth_8), 1)
        return result_depth


    def eight_way_propagation(self, weight_matrix, blur_matrix, kernel):
        #TODO: what about weighing the center pixel itself?...each pixel is replaced by a weighted sum of it's surrounding excluding itself...why not add another center_center_pad(blurred_depth) and get the number of sum elements to 9???
        ### as i understand it, the 3D convolution is only a tool to efficiently get average filter weighted by affinity.... maybe Pac-Convs can replace this?!?@!? ###
        # print(weight_matrix.shape)
        # print(blur_matrix.shape)
        ### Averaging Kernel used to average guidance filters spatially: ###
        sum_conv_weight = torch.ones((1, 8, 1, kernel//2, kernel//2), device=weight_matrix.device)
        ### Multiply blurred depth by weight matrix and 3D-Conv everything with 3D averaging kernel to make one propagation/refinement: ###
        _total_sum = F.conv3d(weight_matrix * blur_matrix, sum_conv_weight)
        ### 3D Conv weight matrix itself: ###
        _weight_sum = F.conv3d(weight_matrix, sum_conv_weight)

        ### Normalize Output by dividing resultant blurred depth by weights: ###
        epsilon = 1e-4
        out = torch.div(_total_sum, torch.abs(_weight_sum)+epsilon)
        return out








# from network.libs.base.pac import conv2d as PAC_Conv2d  #TODO: understand and get PAC_Conv library
class CSPN_V3(nn.Module):   #builds upon PAC-Conv

    def __init__(self, number_of_refinement_iterations):
        super(CSPN_V3, self).__init__()
        self.number_of_refinement_iterations = number_of_refinement_iterations

    def forward(self, x, guided, sparse_depth=None):  #input 8 guidance maps in the form of 8 channels (assuming we diffuse an affinity of 3x3 neighborhood)
        """
        :param x:        Feature maps, N,C,H,W
        :param guided:   guided Filter, N, K^2-1, H, W, K is kernel size
        :return:         returned feature map, N, C, H, W
        """

        B, C, H, W = guided.size()
        K = int(math.sqrt(C + 1))

        ### Softmax along the affinity of neighboring pixels (TODO: by the way i can see if this helps even in the previous versions!!!!): ###
        guided = F.softmax(guided, dim=1)

        kernel = torch.zeros(B, C + 1, H, W, device=guided.device)
        kernel[:, 0:C // 2, :, :] = guided[:, 0:C // 2, :, :]
        kernel[:, C // 2 + 1:C + 1, :, :] = guided[:, C // 2:C, :, :]

        kernel = kernel.unsqueeze(dim=1).reshape(B, 1, K, K, H, W)

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            _x = x

        for _ in range(self.number_of_refinement_iterations):
            x = PAC_Conv2d(x, kernel, kernel_size=K, stride=1, padding=K // 2, dilation=1)

            if sparse_depth is not None:
                no_sparse_mask = 1 - sparse_mask
                x = sparse_mask * _x + no_sparse_mask * x
        return x
##############################################################################################################################################################################################################################











##############################################################################################################################################################################################################################
class SPN_V1(nn.Module):
    def __init__(self):
        super(SPN_V1, self).__init__()

    def forward(self, guidance, blurred_depth):
        # TODO: this is written horribly -> write more efficient!
        # guidance number of channels should be:  number_of_image_to_refine_channels * 4(directions)

        ### Get Weights for each refinement direction (each weights map is 16 channels according to the refined initial multiscale feature extractor: ###
        B, C, H, W = guidance.shape;
        four_directions = C // 4;
        conv4_bn_x1 = conv9[:, 0 * four_directions: 1 * four_directions, :, :]
        conv4_bn_y1 = conv9[:, 1 * four_directions: 2 * four_directions, :, :]
        conv4_bn_x2 = conv9[:, 2 * four_directions: 3 * four_directions, :, :]
        conv4_bn_y2 = conv9[:, 3 * four_directions: 4 * four_directions, :, :]

        ### Perform SPN Refinement: ###
        # (1). Left-Right:
        rnn_h1 = ((1 - conv4_bn_x1[:, :, :, 0]) * blurred_depth[:, :, :, 0]).unsqueeze(3)
        rnn_h2 = ((1 - conv4_bn_x1[:, :, :, 0]) * blurred_depth[:, :, :, W - 1]).unsqueeze(3)
        for i in range(1, W):
            rnn_h1_t = conv4_bn_x1[:, :, :, i] * rnn_h1[:, :, :, i - 1] + (1 - conv4_bn_x1[:, :, :, i]) * blurred_depth[:, :, :, i]
            rnn_h2_t = conv4_bn_x1[:, :, :, i] * rnn_h2[:, :, :, i - 1] + (1 - conv4_bn_x1[:, :, :, i]) * blurred_depth[:, :, :, W - i - 1]
            rnn_h1 = torch.cat([rnn_h1, rnn_h1_t.unsqueeze(3)], dim=3)
            rnn_h2 = torch.cat([rnn_h2, rnn_h2_t.unsqueeze(3)], dim=3)
        # (2). Top-Down:
        rnn_h3 = ((1 - conv4_bn_y1[:, :, 0, :]) * blurred_depth[:, :, 0, :]).unsqueeze(2)
        rnn_h4 = ((1 - conv4_bn_y1[:, :, 0, :]) * blurred_depth[:, :, H - 1, :]).unsqueeze(2)
        for i in range(1, H):
            rnn_h3_t = conv4_bn_x1[:, :, i, :] * rnn_h3[:, :, i - 1, :] + (1 - conv4_bn_x1[:, :, i, :]) * blurred_depth[:, :, i, :]
            rnn_h4_t = conv4_bn_x1[:, :, i, :] * rnn_h4[:, :, i - 1, :] + (1 - conv4_bn_x1[:, :, i, :]) * blurred_depth[:, :, H - i - 1, :]
            rnn_h3 = torch.cat([rnn_h3, rnn_h3_t.unsqueeze(2)], dim=2)
            rnn_h4 = torch.cat([rnn_h4, rnn_h4_t.unsqueeze(2)], dim=2)
        # (3). Left-Right 2nd Pass:
        rnn_h5 = ((1 - conv4_bn_x2[:, :, :, 0]) * rnn_h1[:, :, :, 0]).unsqueeze(3)
        rnn_h6 = ((1 - conv4_bn_x2[:, :, :, 0]) * rnn_h2[:, :, :, W - 1]).unsqueeze(3)
        for i in range(1, W):
            rnn_h5_t = conv4_bn_x2[:, :, :, i] * rnn_h5[:, :, :, i - 1] + (1 - conv4_bn_x2[:, :, :, i]) * rnn_h1[:, :, :, i]
            rnn_h6_t = conv4_bn_x2[:, :, :, i] * rnn_h6[:, :, :, i - 1] + (1 - conv4_bn_x2[:, :, :, i]) * rnn_h2[:, :, :, W - i - 1]
            rnn_h5 = torch.cat([rnn_h5, rnn_h5_t.unsqueeze(3)], dim=3)
            rnn_h6 = torch.cat([rnn_h6, rnn_h6_t.unsqueeze(3)], dim=3)
        # (4). Top-Bottom 2nd Pass:
        rnn_h7 = ((1 - conv4_bn_y2[:, :, 0, :]) * rnn_h3[:, :, 0, :]).unsqueeze(2)
        rnn_h8 = ((1 - conv4_bn_y2[:, :, 0, :]) * rnn_h4[:, :, H - 1, :]).unsqueeze(2)
        for i in range(1, H):
            rnn_h7_t = conv4_bn_y2[:, :, i, :] * rnn_h7[:, :, i - 1, :] + (1 - conv4_bn_y2[:, :, i, :]) * rnn_h3[:, :, i, :]
            rnn_h8_t = conv4_bn_y2[:, :, i, :] * rnn_h8[:, :, i - 1, :] + (1 - conv4_bn_y2[:, :, i, :]) * rnn_h4[:, :, H - i - 1, :]
            rnn_h7 = torch.cat([rnn_h7, rnn_h7_t.unsqueeze(2)], dim=2)
            rnn_h8 = torch.cat([rnn_h8, rnn_h8_t.unsqueeze(2)], dim=2)
        # (5). Max-Out Aggregation -> Convs:
        concat6 = torch.cat([rnn_h5.unsqueeze(4), rnn_h6.unsqueeze(4), rnn_h7.unsqueeze(4), rnn_h8.unsqueeze(4)], dim=4)
        Refined_input = torch.max(concat6, dim=4)[0]

        return Refined_input





class SPN_V2(nn.Module):
    def __init__(self):
        super(SPN_V2, self).__init__()

    ### Get Context From The Immediate Previous Trio Of Pixels: ###
    def to_tridiagonal_multidim(self, w):
        N, W, C, D = w.size()
        tmp_w = w / torch.sum(torch.abs(w), dim=3).unsqueeze(-1)
        tmp_w = tmp_w.unsqueeze(2).expand([N, W, W, C, D])

        eye_a = Variable(torch.diag(torch.ones(W - 1).cuda(), diagonal=-1))
        eye_b = Variable(torch.diag(torch.ones(W).cuda(), diagonal=0))
        eye_c = Variable(torch.diag(torch.ones(W - 1).cuda(), diagonal=1))

        tmp_eye_a = eye_a.unsqueeze(-1).unsqueeze(0).expand([N, W, W, C])
        a = tmp_w[:, :, :, :, 0] * tmp_eye_a
        tmp_eye_b = eye_b.unsqueeze(-1).unsqueeze(0).expand([N, W, W, C])
        b = tmp_w[:, :, :, :, 1] * tmp_eye_b
        tmp_eye_c = eye_c.unsqueeze(-1).unsqueeze(0).expand([N, W, W, C])
        c = tmp_w[:, :, :, :, 2] * tmp_eye_c
        return a + b + c

    def forward(self, guidance, blurred_depth):
        # TODO: this is written horribly -> write more efficient!
        # guidance number of channels should be: number_of_image_to_refine_channels * 4(directions) * 3(neighboring pixels)

        ### permute guidance indices: ###
        N, C, H, W = guidance.size()
        four_directions = C // 4
        C_base = four_directions // 3
        guidance_reshaped_W = guidance.permute(0, 2, 3, 1)  # reshape to make it more "comfortable" for their script - should be changed to avoid wasting time permuting
        x_t = coarse_segmentation.permute(0, 2, 3, 1)

        ### Extract guidance for the 4 directions (left-right, right-left, top-bottom, bottom-up): ###
        conv_x1_flat = guidance_reshaped_W[:, :, :, 0 * four_directions: 1 * four_directions].contiguous()
        conv_y1_flat = guidance_reshaped_W[:, :, :, 1 * four_directions: 2 * four_directions].contiguous()
        conv_x2_flat = guidance_reshaped_W[:, :, :, 2 * four_directions: 3 * four_directions].contiguous()
        conv_y2_flat = guidance_reshaped_W[:, :, :, 3 * four_directions: 4 * four_directions].contiguous()

        ### Change dimensions as preperation to using 3 "behind" pixels instead of 1 instead of 1: ###
        w_x1 = conv_x1_flat.view(N, H, W, four_directions // 3, 3)  # N, H, W, C, 3
        w_y1 = conv_y1_flat.view(N, H, W, four_directions // 3, 3)  # N, H, W, C, 3
        w_x2 = conv_x2_flat.view(N, H, W, four_directions // 3, 3)  # N, H, W, C, 3
        w_y2 = conv_y2_flat.view(N, H, W, four_directions // 3, 3)  # N, H, W, C, 3

        ### Initialize results: ###
        rnn_h1 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())
        rnn_h2 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())
        rnn_h3 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())
        rnn_h4 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())

        ### Horizontal: ###
        for i in range(W):
            ### left to right: ###
            tmp_w = w_x1[:, :, i, :, :]  # N, H, 1, C, 3
            tmp_w = self.to_tridiagonal_multidim(tmp_w)  # N, H, W, C
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h1[:, :, i - 1, :].clone().unsqueeze(1).expand([N, W, H, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t[:, :, i, :]
            rnn_h1[:, :, i, :] = w_x_curr + w_h_prev

            ### right to left: ###
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h2[:, :, W - i, :].clone().unsqueeze(1).expand([N, W, H, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t[:, :, W - i - 1, :]
            rnn_h2[:, :, W - i - 1, :] = w_x_curr + w_h_prev

        ### Vertical: ###
        w_y1_T = w_y1.transpose(1, 2)
        x_t_T = x_t.transpose(1, 2)
        for i in range(H):
            ### up to down: ###
            tmp_w = w_y1_T[:, :, i, :, :]  # N, W, 1, C, 3
            tmp_w = self.to_tridiagonal_multidim(tmp_w)  # N, W, H, C
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h3[:, :, i - 1, :].clone().unsqueeze(1).expand([N, H, W, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t_T[:, :, i, :]
            rnn_h3[:, :, i, :] = w_x_curr + w_h_prev

            ### down to up: ###
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h4[:, :, H - i, :].clone().unsqueeze(1).expand([N, H, W, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * x_t[:, :, H - i - 1, :]
            rnn_h4[:, :, H - i - 1, :] = w_x_curr + w_h_prev
        rnn_h3 = rnn_h3.transpose(1, 2)
        rnn_h4 = rnn_h4.transpose(1, 2)

        ### Initialize Results Before Second Pass: ###
        rnn_h5 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())
        rnn_h6 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())
        rnn_h7 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())
        rnn_h8 = Variable(torch.zeros((N, H, W, four_directions // 3)).cuda())

        ### Horizontal 2nd Pass: ###
        for i in range(W):
            ### left to right: ###
            tmp_w = w_x2[:, :, i, :, :]  # N, H, 1, C, 3
            tmp_w = self.to_tridiagonal_multidim(tmp_w)  # N, H, W, C
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h5[:, :, i - 1, :].clone().unsqueeze(1).expand([N, W, H, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h1[:, :, i, :]
            rnn_h5[:, :, i, :] = w_x_curr + w_h_prev

            ### right to left: ###
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h6[:, :, W - i, :].clone().unsqueeze(1).expand([N, W, H, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h2[:, :, W - i - 1, :]
            rnn_h6[:, :, W - i - 1, :] = w_x_curr + w_h_prev

        ### Vertical 2nd Pass: ###
        w_y2_T = w_y2.transpose(1, 2)
        rnn_h3_T = rnn_h3.transpose(1, 2)
        rnn_h4_T = rnn_h4.transpose(1, 2)
        for i in range(H):
            ### up to down: ###
            tmp_w = w_y2_T[:, :, i, :, :]  # N, W, 1, C, 3
            tmp_w = self.to_tridiagonal_multidim(tmp_w)  # N, W, H, C
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h7[:, :, i - 1, :].clone().unsqueeze(1).expand([N, H, W, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h3_T[:, :, i, :]
            rnn_h7[:, :, i, :] = w_x_curr + w_h_prev

            ### down to up: ###
            if i == 0:
                w_h_prev = 0
            else:
                w_h_prev = torch.sum(tmp_w * rnn_h8[:, :, H - i, :].clone().unsqueeze(1).expand([N, H, W, C_base]), dim=2)
            w_x_curr = (1 - torch.sum(tmp_w, dim=2)) * rnn_h4_T[:, :, H - i - 1, :]
            rnn_h8[:, :, H - i - 1, :] = w_x_curr + w_h_prev

        ### Redundant? Mistake?: ###
        rnn_h3 = rnn_h3.transpose(1, 2)
        rnn_h4 = rnn_h4.transpose(1, 2)

        ### Concatenated Results and use MaxOut Strategy: ###
        concat6 = torch.cat([rnn_h5.unsqueeze(4), rnn_h6.unsqueeze(4), rnn_h7.unsqueeze(4), rnn_h8.unsqueeze(4)], dim=4)
        elt_max = torch.max(concat6, dim=4)[0]

        ### ReArrange To [B,C,H,W]: ###
        Refined_input = elt_max.permute(0, 3, 1, 2)

        return Refined_input




















