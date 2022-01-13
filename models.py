import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from lib.nn import SynchronizedBatchNorm2d
import math
from models.init_weights import init_weights
from models.AttenR2UNet import R2AttU_Net
from models.Bio_Net import BiONet
from torch.nn import init
from smoothgrad import generate_smooth_grad
from guided_backprop import GuidedBackprop
from vanilla_backprop import VanillaBackprop
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

from .attention_blocks import DualAttBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def iou(self, imPred, imLab, numClass):
        _, imPred = torch.max(imPred, dim=1)
        imPred = np.asarray(imPred.cpu()).copy()
        imLab = np.asarray(imLab.cpu()).copy()

        imPred += 1
        imLab += 1
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        imPred = imPred * (imLab > 0)

        # Compute area intersection:
        intersection = imPred * (imPred == imLab)
        (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

        # Compute area union:
        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection
        return 1.0 - (area_intersection + 1e-5) / (area_union + 1e-5)
        # jaccard = area_intersection/area_union
        # #print("I: " + str(area_intersection))
        # #print("U: " + str(area_union))
        # jaccard = (jaccard[1]+jaccard[2])/2
        # return jaccard if jaccard <= 1 else 0

    def pixel_acc(self, pred, label, num_class):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10) #When you +falsePos, acc == Jaccard.
        
        jaccard = []
         
        # calculate jaccard for classes indexed 1 to num_class-1.
        for i in range(1, num_class):
            v = (label == i).long()
            pred = (preds == i).long()
            anb = torch.sum(v * pred)
            try:
                j = anb.float() / (torch.sum(v).float() + torch.sum(pred).float() - anb.float() + 1e-10)
            except:
                j = 0

            j = j if j <= 1 else 0
            jaccard.append(j)

        return acc, jaccard

    def jaccard(self, pred, label):
        AnB = torch.sum(pred.long() & label)
        return AnB/(pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)

    def iou(self, pred, label,smooth=1e-5):
        '''
        Returns the intersection-over-union metric between `a' and `b', which is 0 if identical, 1 if `a' is completely
        disjoint from `b', and values in between if there is overlap. Both inputs are thresholded to binary masks.
        '''
        pred = pred == pred.max()
        label = label == label.max()

        inter = pred * label
        union = pred + label

        return 1.0 - (inter.sum() + smooth) / (union.sum() + smooth)


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, crit, unet, num_class):
        super(SegmentationModule, self).__init__()
        self.crit = crit
        self.unet = unet
        self.num_class = num_class

    def forward(self, feed_dict, epoch, *, segSize=None):
        #training
        if segSize is None:
            p = self.unet(feed_dict['image'])[0]    #############################三个输出，p[0]是网络的整体输出。
            # loss = self.crit(p, feed_dict['mask'], epoch=epoch)      ############### DualLoss需要epoch参数，换成交叉熵就不用epoch参数
            loss = self.crit(p, feed_dict['mask'].long().cuda())

            acc = self.pixel_acc(torch.round(nn.functional.softmax(p, dim=1)).long(), feed_dict['mask'].long().cuda(),
                                 self.num_class)
            iou = self.iou(torch.round(nn.functional.softmax(p, dim=1)).long(), feed_dict['mask'].long().cuda(),
                                 self.num_class)
            return loss, acc, iou

        #test   ##### test_and_pack
        if segSize == True:
            p = self.unet(feed_dict['image'])
            pred = nn.functional.softmax(p, dim=1)
            return pred

        #inference   #####eval
        else:
            p = self.unet(feed_dict['image'])
            loss = self.crit(p, feed_dict['mask'].long().cuda().unsqueeze(0))
            pred = nn.functional.softmax(p, dim=1)
            return pred, loss

    def SRP(self, model, pred, seg):
        output = pred #actual output
        _, pred = torch.max(nn.functional.softmax(pred, dim=1), dim=1) #class index prediction

        tmp = []
        for i in range(output.shape[1]):
            tmp.append((pred == i).long().unsqueeze(1))
        T_model = torch.cat(tmp, dim=1).float()

        LRP = self.layer_relevance_prop(model, output * T_model)

        return LRP

    def layer_relevance_prop(self, model, R):
        for l in range(len(model.layers), 0, -1):
            print(model.layers[l-1])
            R  = model.layers[l-1].relprop(R)
        return R


class ModelBuilder():
    # custom weights initialization

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def build_unet(self, num_class=3, arch='albunet', weights=''):  #########原来num_class = 1,现修改3
        arch = arch.lower()

        # if arch == 'saunet':
        #     unet = SAUNet(num_classes=num_class)
        # if arch == 'myunet':
        #     unet = MYUNet(num_classes=num_class)
        # if arch == 'rattunet':
        #     unet = RAttUNet()
        if arch == 'transresnet':
            unet = TransResNet
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            unet.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print("Loaded pretrained UNet weights.")
        print('Loaded weights for unet')
        return unet

##############################AttenR2UNet 3D

import torch
import torch.nn as nn
# from models.basiclayer import RRCNN_block, up_conv, Attention_block
from models.init_weights import *
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2)),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
            # nn.ConvTranspose3d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            # nn.InstanceNorm3d(ch_out),
            # nn.PReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i ==0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            SE(ch_out),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class SE(nn.Module):
    def __init__(self, in_channels):
        super(SE,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.Conv_Squeeze = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.Conv_Excitation = nn.Conv3d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.relu(z)
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv3d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv3d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv3d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv3d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=[1,2, 2], stride=[1,2, 2])
        self.pool2 = nn.MaxPool3d(kernel_size=[1,3, 3], stride=[1,3, 3])
        self.pool3 = nn.MaxPool3d(kernel_size=[1,5, 5], stride=[1,5, 5])

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, d,h, w = x.size(1), x.size(2), x.size(3),x.size(4)
        self.layer1 = nn.functional.interpolate(self.conv(self.pool1(x)), size=(d,h, w))
        self.layer2 = nn.functional.interpolate(self.conv(self.pool2(x)), size=(d,h, w))
        self.layer3 = nn.functional.interpolate(self.conv(self.pool3(x)), size=(d,h, w))
        out = torch.cat([self.layer1, self.layer2, self.layer3, x], 1)

        return out


class RAttUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=3, t=2):
        super(RAttUNet, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=16, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=16, ch_out=32, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=32, ch_out=64, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.center1 = DACblock(256)
        self.center2 = SPPblock(256)

        self.Up5 = up_conv(ch_in=259, ch_out=128)
        self.Att5 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN5 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN4 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_RRCNN3 = RRCNN_block(ch_in=64, ch_out=32, t=t)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Att2 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.Up_RRCNN2 = RRCNN_block(ch_in=32, ch_out=16, t=t)

        self.Conv_1x1 = nn.Conv3d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        # print('x1.shape:', x1.shape)
        x2 = self.Maxpool(x1)

        x2 = self.RRCNN2(x2)
        # print('x2.shape:', x2.shape)
        x3 = self.Maxpool(x2)

        x3 = self.RRCNN3(x3)
        # print('x3.shape:', x3.shape)
        x4 = self.Maxpool(x3)

        x4 = self.RRCNN4(x4)
        # print('x4.shape:', x4.shape)
        x5 = self.Maxpool(x4)

        x5 = self.RRCNN5(x5)
        # print('x5.shape:', x5.shape)
        c1 = self.center1(x5)
        # print("c1.shape:",c1.shape)

        c2 = self.center2(c1)
        # print("c2.shape:",c2.shape)
        # decoding + concat path
        d5 = self.Up5(c2)

        # d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)
        # print('d5.shape:', d5.shape)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)
        # print('d4.shape:', d4.shape)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)
        # print('d3.shape:', d3.shape)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)
        # print('d2.shape:', d2.shape)

        d1 = self.Conv_1x1(d2)
        # print('d1.shape:', d1.shape)

        return d1, d2, d3


class RegressionNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2):
        super(RegressionNet, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=16, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=32, ch_out=32, t=t)
        ### Spatial_temporal convolution
        self.conv3d_3 = nn.Sequential(nn.Conv3d(64, 64, kernel_size = (3,1,1), stride = 1, padding = (1,0,0)),
                                      nn.Conv3d(64, 64, kernel_size = (1,3,3), stride = 1, padding = (0,1,1)),
                                      nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.conv3d_4 = nn.Sequential(nn.Conv3d(64, 64, kernel_size = (3,1,1), stride = 1, padding = (0,0,0)),
                                      nn.Conv3d(64, 128, kernel_size = (1,3,3), stride = 1, padding = (0,1,1)),
                                      nn.BatchNorm3d(128), nn.ReLU(inplace=True))
        self.conv3d_5 = nn.Sequential(nn.Conv3d(128, 128, kernel_size = (3,1,1), stride = 1, padding = (0,0,0)),
                                      nn.Conv3d(128, 256, kernel_size = (1,3,3), stride = 1, padding = (0,1,1)),
                                      nn.BatchNorm3d(256), nn.ReLU(inplace=True))

        self.pool3 = nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2), padding = (0,0,0))
        self.pool4 = nn.MaxPool3d(kernel_size = (1,2,2), stride = (1,2,2), padding = (0,0,0))
        self.pool5 = nn.AvgPool3d(kernel_size = (1,2,2), stride = (1,2,2), padding = (0,0,0))

        self.conv_reg5 = nn.Conv3d(256, 256, kernel_size = (3, 3, 3), stride = 1, padding = 1)
        self.conv_reg6 = nn.Conv3d(256, 11, kernel_size = (1, 5, 5), stride = 1, padding = 0)

    def forward(self, x, y, z):
        x1 = self.RRCNN1(x)
        # print('x1.shape:', x1.shape)
        x1 = torch.cat((x1, y), 1)
        # print('x11.shape:', x1.shape)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        # print('x2.shape:', x2.shape)
        x3 = torch.cat((x2, z),1)
        # print('x3.shape:',x3.shape)  ###x3.shape: torch.Size([2, 64, 5, 40, 40])
        x4 = self.pool3(self.conv3d_3(x3))
        # print('x4.shape:',x4.shape)  ###x3.shape: torch.Size([2, 64, 5, 40, 40])

        x5 = self.pool4(self.conv3d_4(x4))
        # print('x5.shape:',x5.shape)  ###x3.shape: torch.Size([2, 64, 5, 40, 40])

        x6 = self.pool5(self.conv3d_5(x5))
        # print('x6.shape:',x6.shape)  ###x3.shape: torch.Size([2, 64, 5, 40, 40])

        out_int =x6
        ##### regression
        out = self.conv_reg5(x6)
        # print('out33.shape:', out.shape)
        out_regression = self.conv_reg6(out)
        # print('out3.shape:', out_regression.shape)
        out = out_regression.view(out_regression.size(0), -1)
        return out, out_int


class PhaseClassification(nn.Module):

    def __init__(self):
        super(PhaseClassification, self).__init__()

        self.conv3d_5 = nn.Conv3d(256, 40, kernel_size = (1,5,5), stride = 1, padding = (0,1,1))

        self.batch_norm5 = nn.BatchNorm3d(40)

        self.fc6 = nn.Linear(360,360)
        self.fc7 = nn.Linear(360,2)

    def forward(self, x):

        x = self.conv3d_5(x)
        # print('x1.shape',x.shape)
        x = self.batch_norm5(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        # print('x2.shape',x.shape)

        x = F.relu(self.fc6(x))
        # print('x3.shape',x.shape)

        x = self.fc7(x)

        return x


import math
import copy

from torch.nn.modules.utils import _pair
from collections import OrderedDict
from config import ModelConfig as config
##########Transformer Unet 3D 模型
from collections import OrderedDict
from config import ModelConfig as config
import torch as t
import torch.nn as nn

from torch.autograd import Variable
import torch.nn.functional
# import numpy as np
import warnings
import math

class SingleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        # print('single.shape:',x.shape)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch),
            nn.RReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print('doubleconv.shape:',x.shape)
        return x


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResConvBlock, self).__init__()
        self.conv1 = DoubleConvBlock(in_ch, in_ch)
        self.relu = nn.RReLU(inplace=True)
        self.conv2 = SingleConvBlock(in_ch, out_ch)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x + res)
        x = self.conv2(x)
        # print('Resconv.shape:',x.shape)

        return x


class ConvUpBlock(nn.Module):
    def __init__(self, in_data, out_data, kernel=3, stride=1, padding=1):
        super(ConvUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_data, out_data, kernel_size=3, stride=2, padding=1, output_padding=1,
                               dilation=1),
            nn.GroupNorm(out_data, out_data),
            # nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )
        self.down = nn.Sequential(
            DoubleConvBlock(2 * out_data, out_data),
            nn.RReLU(inplace=True)
        )

    def forward(self, x, down_features):
        x = self.up(x)
        x = t.cat([x, down_features], dim=1)
        x = self.down(x)
        return x


class Attention(nn.Module):
    """
        :params `config.hidden_size`: 256
    """

    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_att_heads = config.transformer__num_heads  # 16
        self.att_head_size = int(config.hidden_size / self.num_att_heads)  # 256 / 16 = 16
        self.all_head_size = self.num_att_heads * self.att_head_size  # 16 * 16 = 256

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def reshape_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.reshape_for_scores(q)
        k = self.reshape_for_scores(k)
        v = self.reshape_for_scores(v)

        qk = t.matmul(q, k.transpose(-1, -2))
        qk = qk / math.sqrt(self.att_head_size)
        qk = self.softmax(qk)

        qkv = t.matmul(qk, v)
        qkv = qkv.permute(0, 2, 1, 3).contiguous()
        new_qkv_shape = qkv.size()[:-2] + (self.all_head_size,)
        qkv = qkv.view(*new_qkv_shape)

        att_output = self.out(qkv)

        return att_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer__mlp_dim)
        self.fc2 = nn.Linear(config.transformer__mlp_dim, config.hidden_size)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(config.transformer__dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


'''
input.shape: torch.Size([1, 4, 192, 192, 48])
resnet.shape: torch.Size([1, 256, 12, 12, 3])
embedding.shape: torch.Size([1, 432, 256])
output.shape: torch.Size([1, 4, 192, 192, 48])
'''


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config

        max_len = 256
        pe = t.zeros(max_len, config.hidden_size)
        position = t.arange(0., max_len).unsqueeze(1)
        div_term = t.exp(t.arange(0., config.hidden_size, 2) * (-math.log(10000.0) / config.hidden_size))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(config.transformer__dropout_rate)

    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention = Attention(config)
        self.mlp = Mlp(config)
        self.att_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        res = x
        x = self.att_norm(x)
        x = self.attention(x)
        x = x + res

        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + res
        return x


class ResNet(nn.Module):
    def __init__(self, in_ch):
        super(ResNet, self).__init__()
        in_kn = [16, 32, 64, 128]
        out_kn = [32, 64, 128, 256]
        self.in_model = nn.Sequential(
            DoubleConvBlock(in_ch, 16),
            nn.RReLU(inplace=True)
        )
        self.body = nn.Sequential(OrderedDict([
            (f'resblock_{i:d}', nn.Sequential(
                ResConvBlock(i_kn, o_kn),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )) for i, (i_kn, o_kn) in enumerate(zip(in_kn, out_kn))
        ]))

    def forward(self, x):
        features = []
        x = self.in_model(x)
        # print('x_inmode.shape:',x.shape)
        features.append(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            # print('x_body.shape:',x.shape)
            features.append(x)
            # print('feature.shape:',features.__sizeof__())
        x = self.body[-1](x)
        # print('x_res.shape:')

        return x, features[::-1]


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1,
                               output_padding=(0, 1, 1),
                               dilation=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch),
            nn.RReLU(inplace=True),
        )
        self.down = nn.Sequential(
            DoubleConvBlock(2 * out_ch, out_ch),
            nn.RReLU(inplace=True),
        )

    def forward(self, x, down_features):
        # print('x_de.shape: down_feature.shape:', x.shape, down_features.shape)
        x = self.up(x)
        # print('x.up.shape:',x.shape)
        x = t.cat([x, down_features], dim=1)
        # print('x.cat.shape:',x.shape)
        x = self.down(x)
        # print('x.down.shape:',x.shape)
        return x


class Decoders(nn.Module):
    def __init__(self, config, trans=True):
        super(Decoders, self).__init__()
        self.trans = trans
        self.conv = nn.Sequential(
            DoubleConvBlock(config.hidden_size, config.hidden_size),
            nn.RReLU(inplace=True)
        )
        # self.conv = SingleConvBlock(config.hidden_size, config.hidden_size)
        in_chs = [256, 128, 64, 32]
        out_chs = [128, 64, 32, 16]
        layers = [
            Decoder(in_ch, out_ch) for in_ch, out_ch in zip(in_chs, out_chs)
        ]
        self.decoders = nn.ModuleList(layers)
        self.out = nn.Sequential(
            DoubleConvBlock(out_chs[-1], 3),  ###4->3
            nn.ReLU(inplace=True)
        )
        # self.out = SingleConvBlock(out_chs[-1], 4)

    def forward(self, x, features):
        if self.trans:
            B, N, hidden = x.size()
            # print('B,N,Hidden:', B,N,hidden)
            k = int(math.pow(N / 5, 1 / 2))
            # print("k.value:", k)
            x = x.permute(0, 2, 1)
            # print('x.permute.shape:',x.shape)
            x = x.contiguous().view(B, hidden, k, k, k)  ######
            # print('x.contiguous.shape:', x.shape)
        x = self.conv(x)
        # print("x.self.conv.shape:",x.shape)
        y = []
        for i, layer in enumerate(self.decoders):
            x = layer(x, features[i])
            # print('x.layer.shape:',x.shape)
            y.append(x)
        # print('y[].shape:',y[2].shape, y[3].shape)
        # print('x.shape:',x.shape)
        x = self.out(x)
        # print('out.shape:', x.shape)
        return x, y[3], y[2]


"""
⬆: Basic Module

⬇: Main Module
    + 3dresnet
    + embedding
    + transformers
    + decoders
"""


class TransResNet(nn.Module):
    def __init__(self, config):
        super(TransResNet, self).__init__()
        self.resnet = ResNet(in_ch=1)  ####### channel 4->1
        self.embedding = Embedding(config)
        self.transformers = nn.ModuleList()
        self.transformers_norm = nn.LayerNorm(config.hidden_size)
        for _ in range(config.transformer__num_layers):
            layer = Transformer(config)
            self.transformers.append(layer)
        self.decoders = Decoders(config)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, features = self.resnet(x)
        # print('x_resnet.shape:.,' ,x.shape)
        x = self.embedding(x)
        # print('x_embedding.shape:', x.shape)
        for transformer in self.transformers:
            x = transformer(x)
            # print('x.transformer.shape:', x.shape)
        x1, x2, x3 = self.decoders(x, features)
        # print('x_decoder.shape:', x1.shape,x2.shape,x3.shape)
        return x1, x2, x3

















