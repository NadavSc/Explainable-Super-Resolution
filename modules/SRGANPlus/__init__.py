import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from modules.SRGANPlus.ops import *


class Generator(nn.Module):

    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=16, act=nn.PReLU(), scale=4):
        super(Generator, self).__init__()

        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=9, BN=False, act=act)

        resblocks = [ResBlock(channels=n_feats, kernel_size=3, act=act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)

        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=False, act=None)

        self.dropout2d = nn.Dropout2d(p=0.2)

        if (scale == 4):
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)

        self.last_conv = conv(in_channel=n_feats, out_channel=img_feat, kernel_size=3, BN=False, act=nn.Tanh())

    def forward(self, x):

        x = self.conv01(x)
        _skip_connection = x

        x = self.body(x)
        x = self.conv02(x)
        x = self.dropout2d(x)

        feat = x + _skip_connection

        x = self.tail(feat)
        x = self.last_conv(x)

        return x, feat


class Discriminator(nn.Module):

    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, act=nn.LeakyReLU(inplace=True), num_of_block=3,
                 patch_size=96):
        super(Discriminator, self).__init__()
        self.act = act

        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=3, BN=False, act=self.act)
        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=False, act=self.act, stride=2)

        body = [
            DiscrimBlock(in_feats=n_feats * (2 ** i), out_feats=n_feats * (2 ** (i + 1)), kernel_size=3, act=self.act)
            for i in range(num_of_block)]
        self.body = nn.Sequential(*body)

        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))

        tail = []

        tail.append(nn.Linear(self.linear_size, 1024))
        tail.append(self.act)
        tail.append(nn.Linear(1024, 1))
        tail.append(nn.Sigmoid())

        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.conv01(x)
        x = self.conv02(x)
        x = self.body(x)
        x = x.view(-1, self.linear_size)
        x = self.tail(x)

        return x


class vgg19(nn.Module):
    def __init__(self, pre_trained=True, require_grad=False):
        super(vgg19, self).__init__()
        self.vgg_feature = models.vgg19(pretrained=pre_trained).features
        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature]
        self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                          'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                          'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
                          'pool3',
                          'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4',
                          'pool4',
                          'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
                          'pool5']

        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x):

        conv1_1 = self.seq_list[0](x)
        relu1_1 = self.seq_list[1](conv1_1)
        conv1_2 = self.seq_list[2](relu1_1)
        relu1_2 = self.seq_list[3](conv1_2)
        pool1 = self.seq_list[4](relu1_2)

        conv2_1 = self.seq_list[5](pool1)
        relu2_1 = self.seq_list[6](conv2_1)
        conv2_2 = self.seq_list[7](relu2_1)
        relu2_2 = self.seq_list[8](conv2_2)
        pool2 = self.seq_list[9](relu2_2)

        conv3_1 = self.seq_list[10](pool2)
        relu3_1 = self.seq_list[11](conv3_1)
        conv3_2 = self.seq_list[12](relu3_1)
        relu3_2 = self.seq_list[13](conv3_2)
        conv3_3 = self.seq_list[14](relu3_2)
        relu3_3 = self.seq_list[15](conv3_3)
        conv3_4 = self.seq_list[16](relu3_3)
        relu3_4 = self.seq_list[17](conv3_4)
        pool3 = self.seq_list[18](relu3_4)

        conv4_1 = self.seq_list[19](pool3)
        relu4_1 = self.seq_list[20](conv4_1)
        conv4_2 = self.seq_list[21](relu4_1)
        relu4_2 = self.seq_list[22](conv4_2)
        conv4_3 = self.seq_list[23](relu4_2)
        relu4_3 = self.seq_list[24](conv4_3)
        conv4_4 = self.seq_list[25](relu4_3)
        relu4_4 = self.seq_list[26](conv4_4)
        pool4 = self.seq_list[27](relu4_4)

        conv5_1 = self.seq_list[28](pool4)
        relu5_1 = self.seq_list[29](conv5_1)
        conv5_2 = self.seq_list[30](relu5_1)
        relu5_2 = self.seq_list[31](conv5_2)
        conv5_3 = self.seq_list[32](relu5_2)
        relu5_3 = self.seq_list[33](conv5_3)
        conv5_4 = self.seq_list[34](relu5_3)
        relu5_4 = self.seq_list[35](conv5_4)
        pool5 = self.seq_list[36](relu5_4)

        vgg_output = namedtuple("vgg_output", self.vgg_layer)

        vgg_list = [conv1_1, relu1_1, conv1_2, relu1_2, pool1,
                    conv2_1, relu2_1, conv2_2, relu2_2, pool2,
                    conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3, conv3_4, relu3_4, pool3,
                    conv4_1, relu4_1, conv4_2, relu4_2, conv4_3, relu4_3, conv4_4, relu4_4, pool4,
                    conv5_1, relu5_1, conv5_2, relu5_2, conv5_3, relu5_3, conv5_4, relu5_4, pool5]

        out = vgg_output(*vgg_list)

        return out


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range=1,
            norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(norm_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(norm_mean) / std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for p in self.parameters():
            p.requires_grad = False


class perceptual_loss(nn.Module):

    def __init__(self, vgg):
        super(perceptual_loss, self).__init__()
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MeanShift(norm_mean=self.normalization_mean, norm_std=self.normalization_std).to(self.device)
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, HR, SR, layer='relu5_4'):
        ## HR and SR should be normalized [0,1]
        hr = self.transform(HR)
        sr = self.transform(SR)

        hr_feat = getattr(self.vgg(hr), layer)
        sr_feat = getattr(self.vgg(sr), layer)

        return self.criterion(hr_feat, sr_feat), hr_feat, sr_feat


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]





