import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init


def conv(in_channels, out_channels, kernel_size, stride, padding):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channels)
    layer = nn.Sequential(convolution, batch_norm)
    return layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def join(ind):
    return lambda x, y: torch.cat( (x,y), ind )

## normalize to unit vectors
def normalize(normals):
    magnitude = torch.pow(normals, 2).sum(1,keepdim=True)
    magnitude = magnitude.sqrt().repeat(1,3,1,1) 
    normed = normals / (magnitude + 1e-6)
    return normed

## channels : list of ints
## kernel_size : int
## padding : int
## stride_fn : fn(channel_index) --> int
## mult=1 if encoder, 2 if decoder
def build_encoder(channels, kernel_size, padding, stride_fn, mult=1):
    layers = []
    sys.stdout.write( '    %3d' % channels[0] )
    for ind in range(len(channels)-1):
        m = 1 if ind == 0 else mult
        in_channels = channels[ind] * m
        out_channels = channels[ind+1]
        stride = stride_fn(ind)
        sys.stdout.write( ' --> %3d' % out_channels )

        if ind < len(channels)-2:
            block = conv(in_channels, out_channels, kernel_size, stride, padding)
        else:
            block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        layers.append(block)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return nn.ModuleList(layers)
