
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
def make_model():
    return GCSR()

def conv_layer(in_channles,out_channels, kernel_size, stride=1, dilation=1, groups=1,padding=1):
    return nn.Conv2d(in_channles,out_channels, kernel_size, stride=1, dilation=dilation, groups=1,padding=padding,bias=True)
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
         
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class Biasic(nn.Module):
   def __init__(self, inchannles, rate=0.5):
        super(Biasic, self).__init__()
        self.split_channels1 = int(inchannles * rate)
        self.remian_channles1 = int(inchannles - self.split_channels1)
        self.split_channles2 = int((inchannles * rate) / 2)
        self.remain_channels2 = int((inchannles * rate) / 2)
        self.c1 = conv_layer(64,64,kernel_size=3,padding=1)
        self.c2=conv_layer(32,32,kernel_size=(1,3),padding=(0,1))

        self.c22 = GhostModule(32,32,kernel_size=3)
        self.c3 = conv_layer(32,32,kernel_size=(1,3),padding=(0,1))
        self.c33=GhostModule(32,32,kernel_size=3)
        self.c66 = GhostModule(32,32,kernel_size=3)
        self.c6 = conv_layer(32,32,kernel_size=(1,3),padding=(0,1))
        self.c7=conv_layer(32,32,kernel_size=(1,3),padding=(0,1))
        self.c77 = GhostModule(32,32,kernel_size=3)
        self.c4 = conv_layer( 4* inchannles, inchannles, 1,padding=0)
        self.relu = nn.LeakyReLU(0.05, inplace=True)


   def forward(self, x):

        out1 = self.relu(self.c1(x))
        d1, r1 = torch.split(out1, (self.split_channels1, self.remian_channles1), dim=1)
        out21 = self.relu(self.c22(d1))
        out22 = self.relu(self.c2(r1))
        out31 = self.relu(self.c33(out21+d1))
        out32 = self.relu(self.c3(out22+r1))
        out41 = self.relu(self.c6(out32+d1))
        out42 = self.relu(self.c66(out31+r1))
        out43 = self.relu(self.c7(out41+d1))
        out44 = self.relu(self.c77(out42+r1))
        out = torch.cat([out21,out22,out31,out32,out41,out42,out43,out44], dim=1)
        out = self.relu(self.c4(out))
        out = out + x
        return out


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor,)
    return sequential(conv, pixel_shuffle)
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
def default_conv(in_channels, out_channels, kernel_size, bias=True,):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class GCSR(nn.Module):
    def __init__(self):
        super(GCSR, self).__init__()
        n_feats = 64
        kernel_size = 3
        scale = 4
        self.D = 5
        self.sub_mean = MeanShift(255)
        self.light1 =Biasic(n_feats)
        self.light2 =Biasic(n_feats)
        self.light3 = Biasic(n_feats)
        self.light4 = Biasic(n_feats)
        self.light5 = Biasic(n_feats)
        self.light6 = Biasic(n_feats)
       
        self.GEF = conv_block(384, 64, kernel_size=1, act_type='lrelu')
        modules_head = [default_conv(3, n_feats, kernel_size)]
        self.add_mean =MeanShift(255, sign=1)
        self.head = nn.Sequential(*modules_head)
        self.tail1 = nn.Upsample(scale_factor= scale,mode='bicubic',align_corners = False)
        self.LR_conv = conv_layer(64, 64, kernel_size=3)
        self.tail  = pixelshuffle_block(n_feats,3,scale)
    def forward(self, x):
        x = self.sub_mean(x)
        s = self.tail1(x)
        x = self.head(x)	
        out1 = self.light1(x)
        out2 = self.light2(out1)
        out3 = self.light3(out2)
        out4 = self.light4(out3)
        out5 = self.light5(out4)
        out6 = self.light6(out5)
        out_f = self.GEF(torch.cat([out1,out2,out3,out4,out5,out6],dim=1))
        out_f = self.LR_conv(out_f) + x
        out = self.tail(out_f)
        out = out+s
        out = self.add_mean(out)
        return out

