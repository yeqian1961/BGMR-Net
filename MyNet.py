import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, relu=False):
        super(BasicConv2d, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channel)]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class GeluConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=False, relu=False):
        super(GeluConv2d, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channel)]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class asyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs


class GIE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GIE, self).__init__()
        self.gelu = nn.GELU()
        self.asyConv = asyConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                               dilation=1, groups=1,
                               padding_mode='zeros', deploy=False)
        self.dilConv = GeluConv2d(in_channel, out_channel, kernel_size=3, dilation=3, padding=3, stride=1, relu=True)
        self.normConv = GeluConv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, relu=True)

        self.conv_cat = GeluConv2d(3 * out_channel, out_channel, kernel_size=3, padding=1, relu=True)
        self.conv_res = GeluConv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.asyConv(x)
        x2 = self.dilConv(x)
        x3 = self.normConv(x)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3), 1))
        x = self.gelu(x_cat + self.conv_res(x))
        return x


class BA(nn.Module):
    def __init__(self, channel):
        super(BA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.con1 = BasicConv2d(512, channel, 1)
        self.con2 = BasicConv2d(320, channel, 1)
        self.con3 = BasicConv2d(128, channel, 1)
        self.con4 = BasicConv2d(64, channel, 1)

        self.upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.upsample6 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.con5 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)
        self.con6 = nn.Conv2d(4 * channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # 1x1 通道缩减
        x1 = self.con1(x1)
        x2 = self.con2(x2)
        x3 = self.con3(x3)
        x4 = self.con4(x4)
        # 相邻层交互
        x1_1 = x1
        x2_1 = self.upsample1(self.upsample(x1)) * x2
        x3_1 = self.upsample2(self.upsample(x2_1)) * x3
        x4_1 = self.upsample3(self.upsample(x3_1)) * x4
        # 相邻层连接
        x2_2 = torch.cat((x2_1, self.upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x4_2 = torch.cat((x4_1, self.upsample6(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.con5(x4_2)
        x = self.con6(x)
        return x


class SPADE(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(hidden_channels, affine=False)
        self.mlp_shared = BasicConv2d(1, hidden_channels, kernel_size=3, padding=1, relu=True)
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, edge):
        # 输入x特征,edge获得到初步边缘
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        edge = F.interpolate(edge, size=x.size()[2:], mode='bicubic')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out

class CR(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CR, self).__init__()
        self.conv1 = BasicConv2d(in_channel, out_channel, 3, padding=1, relu=True)
        self.conv2 = BasicConv2d(out_channel, out_channel, 3, padding=1, relu=True)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = BasicConv2d(inp, mip, kernel_size=1, stride=1, padding=0, relu=True)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class GroupingConv(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(GroupingConv, self).__init__()
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)


class MFDS(nn.Module):
    def __init__(self, channel, N):
        super(MFDS, self).__init__()
        self.sample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.w = nn.Parameter(torch.ones(1))
        self.v = nn.Parameter(torch.ones(1))
        self.gc1 = GroupingConv(channel * 2, channel, N=N)
        self.gc2 = GroupingConv(channel * 2, channel, N=N)
        self.ca1 = CoordAtt(channel, channel)
        self.ca2 = CoordAtt(channel, channel)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.GBR = GeluConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.spade = SPADE(channel, channel)
        self.out = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, xr, fi, xg, x_edge):
        # reverse attention
        xg_dilate = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        xg_r = -1 * (torch.sigmoid(xg_dilate(xg))) + 1  # 64,H/32,W/32
        fi_r = self.GBR(fi)

        n1 = self.groupfusion(xr, self.sample1(xg))  # 128,H/16,W/16
        zt1 = xr + self.gc1(n1)
        r2 = self.groupfusion(xr, self.sample2(xg_r))  # 128,H/16,W/16
        zt2 = xr + self.gc2(r2)

        zt_a = F.interpolate(fi_r, scale_factor=2, mode='bicubic', align_corners=True) - self.ca1(zt1) * self.w
        refine = self.relu(self.bn(zt_a))
        zt_b = refine + self.ca2(zt2) * self.v

        fea = self.spade(zt_b, x_edge)
        map = self.out(fea)
        return fea, map

    def groupfusion(self, xr, xg):
        M = 8
        xr_g = torch.chunk(xr, M, dim=1)
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])
        return torch.cat(foo, 1)


class COD(nn.Module):
    def __init__(self, channel=64, arc='PVTv2-B2'):
        super(COD, self).__init__()
        # pvt
        if arc == 'PVTv2-B2':
            print('--> using PVTv2-B2 right now')
            self.context_encoder = pvt_v2_b2(pretrained=True)
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))
        in_channel_list = [64, 128, 320, 512]

        self.gie = GIE(in_channel_list[3], channel)
        self.cr2 = CR(in_channel_list[2], channel)
        self.cr3 = CR(in_channel_list[1], channel)
        self.cr4 = CR(in_channel_list[0], channel)
        self.ba = BA(channel)

        self.mfds = MFDS(channel, N=[8, 16, 32])

    def forward(self, x):
        # backbone PVT
        endpoints = self.context_encoder.extract_endpoints(x)
        x4 = endpoints['reduction_2']  # 64,H/4,W/4
        x3 = endpoints['reduction_3']  # 128,H/8,W/8
        x2 = endpoints['reduction_4']  # 320,H/16,W/16
        x1 = endpoints['reduction_5']  # 512,H/32,W/32

        x_edge = self.ba(x1, x2, x3, x4)
        x_global = self.gie(x1)
        xr2 = self.cr2(x2)
        xr3 = self.cr3(x3)
        xr4 = self.cr4(x4)

        f2, y2 = self.mfds(xr2, x_global, x_global, x_edge)
        f3, y3 = self.mfds(xr3, f2, y2 + F.interpolate(x_global, scale_factor=2, mode='bicubic', align_corners=True),
                           x_edge)
        f4, y4 = self.mfds(xr4, f3, y3 + F.interpolate(x_global, scale_factor=4, mode='bicubic', align_corners=True),
                           x_edge)

        edge = F.interpolate(x_edge, scale_factor=4, mode='bicubic', align_corners=True)
        cam_out_2 = F.interpolate(y2, scale_factor=16, mode='bicubic', align_corners=True)
        cam_out_3 = F.interpolate(y3, scale_factor=8, mode='bicubic', align_corners=True)

        # final pred
        cam_out_4 = F.interpolate(y4, scale_factor=4, mode='bicubic', align_corners=True)

        return edge, cam_out_2, cam_out_3, cam_out_4


if __name__ == '__main__':
    import torch

    with torch.cuda.device(0):
        pytorch_model = COD(channel=64, arc='PVTv2-B2').cuda()
        input = torch.randn(1, 3, 352, 352).cuda()
        edge, cam_out_2, cam_out_3, cam_out_4 = pytorch_model(input)
        print(edge.size(), cam_out_2.size(), cam_out_3.size(), cam_out_4.size())

