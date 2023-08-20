import torch
import torch.nn as nn
# from RapidBase.Layers.Conv_Blocks import input_adaptive_ConvNd, input_adaptive_linear

class upsample_and_concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_and_concat, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=3,
             stride=2,
             padding=1)

    def forward(self, x1, x2):
        deconv = self.transpose_conv(x1)
        deconv = torch.cat([deconv, x2], 1)
        return deconv


# def upsample_and_concat(x1, x2, output_channels, in_channels):
#     pool_size = 2
#     deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
#     deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
#
#     deconv_output = tf.concat([deconv, x2], 3)
#     deconv_output.set_shape([None, None, None, output_channels * 2])
#
#     return deconv_output


from RapidBase.Layers.Conv_Blocks import Conv_Block
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1_1 = nn.Conv2d(3, 32, 3)
        self.conv1_2 = nn.Conv2d(32, 32, 3)
        self.conv1_3 = nn.Conv2d(32, 32, 3)
        self.conv1_4 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.conv2_3 = nn.Conv2d(64, 64, 3)
        self.conv2_4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(128, 128, 3)
        self.conv3_3 = nn.Conv2d(128, 128, 3)
        self.conv3_4 = nn.Conv2d(128, 128, 3)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3)
        self.conv4_2 = nn.Conv2d(256, 256, 3)
        self.conv4_3 = nn.Conv2d(256, 256, 3)
        self.conv4_4 = nn.Conv2d(256, 256, 3)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        self.conv5_4 = nn.Conv2d(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2)

        self.up6 = upsample_and_concat(512, 256)
        self.conv6_1 = nn.Conv2d(256, 256, 3)
        self.conv6_2 = nn.Conv2d(256, 256, 3)
        self.conv6_3 = nn.Conv2d(256, 256, 3)

        self.up7 = upsample_and_concat(256, 128)
        self.conv7_1 = nn.Conv2d(128, 128, 3)
        self.conv7_2 = nn.Conv2d(128, 128, 3)
        self.conv7_3 = nn.Conv2d(128, 128, 3)

        self.up8 = upsample_and_concat(128, 64)
        self.conv8_1 = nn.Conv2d(64, 64, 3)
        self.conv8_2 = nn.Conv2d(64, 64, 3)
        self.conv8_3 = nn.Conv2d(64, 64, 3)

        self.up9 = upsample_and_concat(64, 32)
        self.conv9_1 = nn.Conv2d(32, 32, 3)
        self.conv9_2 = nn.Conv2d(32, 32, 3)
        self.conv9_3 = nn.Conv2d(32, 32, 3)

        self.conv10 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1_1(x))
        x1 = self.lrelu(self.conv1_2(x1))
        x1 = self.lrelu(self.conv1_3(x1))
        x1 = self.lrelu(self.conv1_4(x1))
        x1_pool = self.pool1(x1)

        x2 = self.lrelu(self.conv2_1(x1_pool))
        x2 = self.lrelu(self.conv2_2(x2))
        x2 = self.lrelu(self.conv2_3(x2))
        x2 = self.lrelu(self.conv2_4(x2))
        x2_pool = self.pool2(x2)

        x3 = self.lrelu(self.conv3_1(x2_pool))
        x3 = self.lrelu(self.conv3_2(x3))
        x3 = self.lrelu(self.conv3_3(x3))
        x3 = self.lrelu(self.conv3_4(x3))
        x3_pool = self.pool3(x3)

        x4 = self.lrelu(self.conv4_1(x3_pool))
        x4 = self.lrelu(self.conv4_2(x4))
        x4 = self.lrelu(self.conv4_3(x4))
        x4 = self.lrelu(self.conv4_4(x4))
        x4_pool = self.pool4(x4)

        x5 = self.lrelu(self.conv5_1(x4_pool))
        x5 = self.lrelu(self.conv5_2(x5))
        x5 = self.lrelu(self.conv5_3(x5))
        x5 = self.lrelu(self.conv5_4(x5))

        up4 = self.up6(x5, x4)
        up4 = self.lrelu(self.conv6_1(up4))
        up4 = self.lrelu(self.conv6_2(up4))
        up4 = self.lrelu(self.conv6_3(up4))

        up3 = self.up7(up4, x3)
        up3 = self.lrelu(self.conv7_1(up3))
        up3 = self.lrelu(self.conv7_2(up3))
        up3 = self.lrelu(self.conv7_3(up3))

        up2 = self.up8(up3, x2)
        up2 = self.lrelu(self.conv8_1(up2))
        up2 = self.lrelu(self.conv8_2(up2))
        up2 = self.lrelu(self.conv8_3(up2))

        up1 = self.up8(up2, x1)
        up1 = self.lrelu(self.conv9_1(up1))
        up1 = self.lrelu(self.conv9_2(up1))
        up1 = self.lrelu(self.conv9_3(up1))

        output = self.conv10(up1)
        return output


class feature_encoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(feature_encoding, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv4_SE = squeeze_excitation_layer(32, 2)
        self.final = nn.Conv2d(32, out_channels, 3)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.conv4_SE(x)
        x = self.final(x)
        return x


class avg_pool(nn.Module):
    def __init__(self):
        super(avg_pool, self).__init__()
        self.pool1 = nn.AvgPool2d(1, 1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(4, 4)
        self.pool4 = nn.AvgPool2d(8, 8)
        self.pool5 = nn.AvgPool2d(16, 16)

    def forward(self, x):
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        pool4 = self.pool4(x)
        pool5 = self.pool5(x)
        return pool1, pool2, pool3, pool4, pool5


class all_unet(nn.Module):
    def __init__(self):
        super(all_unet, self).__init__()
        self.unet1 = unet()
        self.unet2 = unet()
        self.unet3 = unet()
        self.unet4 = unet()
        self.unet5 = unet()

    def forward(self, pool1, pool2, pool3, pool4, pool5):
        unet1 = self.unet1(pool1)
        unet2 = self.unet2(pool2)
        unet3 = self.unet3(pool3)
        unet4 = self.unet4(pool4)
        unet5 = self.unet5(pool5)
        return unet1, unet2, unet3, unet4, unet5


class resize_all_image(nn.Module):
    def __init__(self):
        super(resize_all_image, self).__init__()
        self.resize = None

    def forward(self, unet1, unet2, unet3, unet4, unet5):
        if self.resize is None:
            self.resize = nn.Upsample(size=(unet1.shape[2], unet1.shape[3]), mode='bilinear')
        resize1 = self.resize(unet1)
        resize2 = self.resize(unet2)
        resize3 = self.resize(unet3)
        resize4 = self.resize(unet4)
        resize5 = self.resize(unet5)
        return resize1, resize2, resize3, resize4, resize5


class to_clean_image(nn.Module):
    def __init__(self):
        super(to_clean_image, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        input_channels = 3
        self.sk_conv1 = nn.Conv2d(input_channels, 7, 3)
        self.sk_conv2 = nn.Conv2d(input_channels, 7, 5)
        self.sk_conv3 = nn.Conv2d(input_channels, 7, 7)
        self.sk_out = selective_kernel_layer(4, 7)
        output_channels = 3
        self.final_conv = nn.Conv2d(output_channels, 1, 3)

    def forward(self, feature_map, resize1, resize2, resize3, resize4, resize5):
        concat = torch.cat([feature_map, resize1, resize2, resize3, resize4, resize5], 1)
        sk_conv1 = self.lrelu(self.sk_conv1(concat))
        sk_conv2 = self.lrelu(self.sk_conv2(concat))
        sk_conv3 = self.lrelu(self.sk_conv3(concat))
        sk_out = self.sk_out(sk_conv1, sk_conv2, sk_conv3)
        output = self.final_conv(sk_out)
        return output


class squeeze_excitation_layer(nn.Module):
    def __init__(self, middle, outdim):
        super(squeeze_excitation_layer, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(32, middle)
        self.dense2 = nn.Linear(middle, outdim)
        self.out_dim = outdim
    def forward(self, x):
        squeeze = self.squeeze(x)
        excitation = self.dense1(x)
        excitation = self.relu(excitation)
        excitation = self.dense2(excitation)
        excitation = self.sigmoid(excitation)
        excitation = torch.reshape(excitation, [-1, self.out_dim, 1, 1])
        scale = x * excitation
        return scale


class selective_kernel_layer(nn.Module):
    def __init__(self, middle, out_dim):
        super(selective_kernel_layer, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.after_softmax = nn.Softmax()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        in_channels = 3
        self.dense1 = nn.Linear(in_channels, middle)
        self.dense2 = nn.Linear(middle, out_dim)
        self.dense3 = nn.Linear(middle, out_dim)
        self.dense4 = nn.Linear(middle, out_dim)
        self.out_dim = out_dim

    def forward(self, sk_conv1, sk_conv2, sk_conv3):
        sum_u = sk_conv1 + sk_conv2 + sk_conv3
        squeeze = self.squeeze(sum_u)
        squeeze = torch.reshape(squeeze, [-1, self.out_dim, 1, 1])
        z = self.relu(self.dense1(squeeze))
        a1 = self.dense2(z)
        a2 = self.dense2(z)
        a3 = self.dense2(z)

        before_softmax = torch.cat([a1,a2,a3], 1)
        after_softmax = torch.softmax(before_softmax, 1)
        a1 = after_softmax[:,0:1, :, :]
        a2 = after_softmax[:,1:2, :, :]
        a3 = after_softmax[:,2:3, :, :]
        select_1 = sk_conv1 * a1
        select_2 = sk_conv2 * a2
        select_3 = sk_conv3 * a3
        out = select_1 + select_2 + select_3
        return out


class PRIDNet(nn.Module):
    def __init__(self, in_channels=3):
        super(PRIDNet, self).__init__()
        self.feature_encoding = feature_encoding(in_channels, 1)
        self.avg_pool = avg_pool()
        self.all_unet = all_unet()
        self.resize_all = resize_all_image()
        self.to_clean_image = to_clean_image()

    def forward(self, x):
        feature_map = self.feature_encoding(x)
        feature_map_2 = torch.cat([x, feature_map], 1)
        pool1, pool2, pool3, pool4, pool5 = self.avg_pool(feature_map_2)
        unet1, unet2, unet3, unet4, unet5 = self.all_unet(pool1, pool2, pool3, pool4, pool5)
        resize1, resize2, resize3, resize4, resize5 = self.resize_all_image(unet1, unet2, unet3, unet4, unet5)
        out_image = self.to_clean_image(feature_map_2, resize1, resize2, resize3, resize4, resize5)
        return out_image

