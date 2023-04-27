"""
## ECCV 2022
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out



class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)
        # import pdb
        # pdb.set_trace()  

        return out



class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc,nc//2)

    def forward(self, x):
        yy=self.block(x)

        return x+yy



class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self,x):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockAdjust(nn.Module):
    def __init__(self, nc):
        super(FreBlockAdjust, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.sft = SFT(nc)
        self.cat = nn.Conv2d(2*nc,nc,1,1,0)

    def forward(self,x, y_amp, y_phase):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        mag = self.sft(mag, y_amp)
        pha = self.cat(torch.cat([y_phase,pha],1))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class ProcessBlock(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlock,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlock(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.contrast = stdv_channels
        # self.process = nn.Sequential(nn.Conv2d(in_nc * 2, in_nc // 2, kernel_size=3, padding=1, bias=True),
        #                              nn.LeakyReLU(0.1),
        #                              nn.Conv2d(in_nc // 2, in_nc * 2, kernel_size=3, padding=1, bias=True),
        #                              nn.Sigmoid())

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class ProcessBlockAdjust(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlockAdjust,self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockAdjust(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x, y_amp, y_phase):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq, y_amp, y_phase)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori


class SFT(nn.Module):
    def __init__(self, nc):
        super(SFT,self).__init__()
        self.convmul = nn.Conv2d(nc,nc,3,1,1)
        self.convadd = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convfuse = nn.Conv2d(2*nc, nc, 1, 1, 0)

    def forward(self, x, res):
        # res = res.detach()
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(torch.cat([x,mul*x+add],1))
        return fuse


def coeff_apply(InputTensor, CoeffTensor, isoffset=True):
    if not isoffset:
        raise ValueError("No-offset is not implemented.")
    bIn, cIn, hIn, wIn = InputTensor.shape
    bCo, cCo, hCo, wCo = CoeffTensor.shape
    assert hIn == hCo and wIn == wCo, 'Wrong dimension: In:%dx%d != Co:%dx%d' % (hIn, wIn, hCo, wCo)
    if isoffset:
        assert cCo % (cIn + 1) == 0, 'The dimension of Coeff and Input is mismatching with offset.'
        cOut = cCo / (cIn + 1)
    else:
        assert cCo % cIn == 0, 'The dimension of Coeff and Input is mismatching without offset.'
        cOut = cCo / cIn
    outList = []

    if isoffset:
        for i in range(int(cOut)):
            Oc = CoeffTensor[:, cIn + (cIn + 1) * i:cIn + (cIn + 1) * i + 1, :, :]
            Oc = Oc + torch.sum(CoeffTensor[:, (cIn + 1) * i:(cIn + 1) * i + cIn, :, :] * InputTensor,
                            dim=1, keepdim=True)
            outList.append(Oc)

    return torch.cat(outList, dim=1)


class HighNet(nn.Module):
    def __init__(self, nc):
        super(HighNet,self).__init__()
        self.conv0 = nn.PixelUnshuffle(2)
        self.conv1 = ProcessBlockAdjust(12)
        # self.conv2 = ProcessBlockAdjust(nc)
        self.conv3 = ProcessBlock(12)
        self.conv4 = ProcessBlock(12)
        self.conv5 = nn.PixelShuffle(2)
        self.convout = nn.Conv2d(3, 3, 3, 1, 1)
        self.trans = nn.Conv2d(6,32,1,1,0)
        self.con_temp1 = nn.Conv2d(32,32,3,1,1)
        self.con_temp2 = nn.Conv2d(32,32,3,1,1)
        self.con_temp3 = nn.Conv2d(32,3,3,1,1)
        self.LeakyReLU=nn.LeakyReLU(0.1, inplace=False)
    def forward(self,x, y_down, y_down_amp, y_down_phase):
        x_ori = x
        x = self.conv0(x) #3*4=12

        x1 = self.conv1(x, y_down_amp, y_down_phase)
        # x2 = self.conv2(x1, y_down_amp, y_down_phase)

        x3 = self.conv3(x1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        xout_temp = self.convout(x5)
        y_aff = self.trans(torch.cat([F.interpolate(y_down, scale_factor=2, mode='bilinear'), xout_temp], 1))
        con_temp1=self.con_temp1(y_aff)
        con_temp2=self.con_temp2(con_temp1)
        xout=self.con_temp3(con_temp2)
        #xout = coeff_apply(x_ori, y_aff)+xout

        return xout


class LowNet(nn.Module):
    def __init__(self, in_nc, nc):
        super(LowNet,self).__init__()
        self.conv0 = nn.Conv2d(in_nc,nc,1,1,0)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.conv2 = ProcessBlock(nc*2)
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.conv3 = ProcessBlock(nc*3)
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.conv4 = ProcessBlock(nc*2)
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc,12,1,1,0)
        self.convoutfinal = nn.Conv2d(12, 3, 1, 1, 0)

        self.transamp = nn.Conv2d(12,12,1,1,0)
        self.transpha = nn.Conv2d(12,12, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)
        xout_fre =  torch.fft.rfft2(xout, norm='backward')
        xout_fre_amp,xout_fre_phase = torch.abs(xout_fre), torch.angle(xout_fre)
        xfinal = self.convoutfinal(xout)

        return xfinal,self.transamp(xout_fre_amp),self.transpha(xout_fre_phase)



class InteractNet(nn.Module):
    def __init__(self, nc=16):
        super(InteractNet,self).__init__()
        self.extract =  nn.Conv2d(3,nc//2,1,1,0)
        self.lownet = LowNet(nc//2, nc*12)
        self.highnet = HighNet(nc)

    def forward(self, x):

        x_f = self.extract(x)
        x_f_down = F.interpolate(x_f,scale_factor=0.5,mode='bilinear')
        y_down, y_down_amp, y_down_phase = self.lownet(x_f_down)
        y = self.highnet(x,y_down, y_down_amp,y_down_phase)

        return y, y_down