import sys, torch, torch.nn as nn, torch.nn.functional as F,torch.optim as optim
from torch.autograd import Variable
from torch import nn


def conv(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bn=True):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
    if bn:
        batch_norm = nn.BatchNorm2d(out_channels)
        layer = nn.Sequential(convolution, batch_norm)
    else:
        instance_norm = nn.InstanceNorm2d(out_channels)
        layer = nn.Sequential(convolution, instance_norm)
    return layer

def build_encoder(channels, kernel_size, padding, stride_fn, dilation=None, mult=1, se_squeeze=None, reduction=None, bn=True):
    layers = []
    sys.stdout.write( '    %3d' % channels[0] )
    for ind in range(len(channels)-1):
        m = 1 if ind == 0 else mult
        if se_squeeze and m == 2:
            in_channels = channels[ind] + channels[ind] // reduction
        else:
            in_channels = channels[ind] * m
        out_channels = channels[ind+1]
        stride = stride_fn(ind)
        sys.stdout.write( ' --> %3d' % out_channels )

        if ind < len(channels)-2:
            block = conv(in_channels, out_channels, kernel_size, stride, padding=dilation[ind] if dilation else 1, dilation=dilation[ind] if dilation else 1, bn=bn)
        else:
            # print(dilation[ind] if dilation else 1)
            block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation[ind] if dilation else 1, dilation=dilation[ind] if dilation else 1)

        layers.append(block)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return nn.ModuleList(layers)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8, act='relu'):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True) if act == 'relu' else nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayerSqueeze(nn.Module):
    def __init__(self, channel, reduction=8, act='relu'):
        super(SELayerSqueeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True) if act == 'relu' else nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(channel, channel // reduction, 1, 1, 0)
        self.act = nn.LeakyReLU(inplace=True) if act == 'relu' else nn.ELU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv1x1(x)
        x = self.act(x)
        return x

class SELayerSqueezeFixed(nn.Module):
    def __init__(self, channel, reduction=8, fixed_value=8):
        super(SELayerSqueezeFixed, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(channel, fixed_value, 1, 1, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv1x1(x)
        return x

class SELayerImproved(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayerImproved, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, bias=True)
        self.fc_list = []
        for _ in range(2):
            self.fc_list.append(nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            ))
        self.fc_list = nn.ModuleList(self.fc_list)

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc_list[0](y1).view(b, c, 1, 1)
        y2 = self.avg_pool(x).view(b, c)
        y2 = self.fc_list[1](y2).view(b, c, 1, 1)
        y1 = x * y1.expand_as(x)
        y2 = x * y2.expand_as(x)
        return F.leaky_relu(self.conv1x1(torch.cat( [y1, y2], 1)), inplace=True)

class SEDecomposerSingle(nn.Module):

    def __init__(self, channels=[3, 32, 64, 128, 256], kernel_size=3, padding=1, skip_se=False, low_se=False, se_improved=False, multi_size=False, image_size=256, se_squeeze=False, reduction=8, detach=False, bn=True, act="relu"):
        super(SEDecomposerSingle, self).__init__()

        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder1 = build_encoder(channels, kernel_size, padding, stride_fn, bn=bn)
        self.skip_se = skip_se
        self.low_se = low_se
        self.se_improved = se_improved
        self.multi_size = multi_size
        self.image_size = image_size
        self.se_squeeze = se_squeeze
        self.reduction = reduction
        self.detach = detach
        self.bn = bn
        if not self.bn:
            print(" IN mode")
        else:
            print(" BN mode")
        print(act)
        self.act = act
        if self.skip_se:
            sys.stdout.write( 'skip SELayer on\n' )
            self.se_skip_layers = []
            for c in channels[1:]:
                if self.se_improved:
                    self.se_skip_layers.append(SELayerImproved(channel=c, reduction=self.reduction))
                elif self.se_squeeze:
                    self.se_skip_layers.append(SELayerSqueeze(channel=c, reduction=self.reduction, act=self.act))
                else:
                    self.se_skip_layers.append(SELayer(channel=c, reduction=self.reduction, act=self.act))
            self.se_skip_modulelist = nn.ModuleList(self.se_skip_layers)
        else:
            sys.stdout.write( 'skip SELayer off\n' )
        if self.se_improved:
            sys.stdout.write( 'SELayer improve on\n' )
        else:
            sys.stdout.write( 'SELayer improve off\n' )
        if self.se_squeeze:
            sys.stdout.write( 'se_squeeze on\n' )
        ## link encoder and decoder
        channels.append(channels[-1])
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Decomposer> Building Decoder' )
        
        self.decoder_reflectance = build_encoder(channels, kernel_size, padding, stride_fn, mult=2, se_squeeze=self.se_squeeze, reduction=self.reduction, bn=self.bn)

        self.upsampler = nn.Upsample(scale_factor=2)
        if self.low_se:
            self.se_layer = SELayerImproved(channels[0]) if self.se_improved else SELayer(channels[0], act=self.act)
        if self.multi_size:
            if self.se_squeeze:
                self.frame1 = nn.Conv2d(128 + 128 // self.reduction, 3, 3, 1, 1)
                self.frame2 = nn.Conv2d(64 + 64 // self.reduction, 3, 3, 1, 1)
            else:
                self.frame1 = nn.Conv2d(256, 3, 3, 1, 1)
                self.frame2 = nn.Conv2d(128, 3, 3, 1, 1)

    def __decode(self, decoder, encoded, inp):
        x = inp
        frame_list = []
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            if self.skip_se:
                encoded[-(ind+1)] = self.se_skip_modulelist[-(ind+1)](encoded[-(ind+1)])
            x = torch.cat( (x, encoded[-(ind+1)]), 1)
            if self.act == 'relu':
                x = F.leaky_relu(x, inplace=True)
            else:
                x = F.elu(x, inplace=True)
            if self.multi_size:
                if x.size()[-1] == self.image_size // 4 and x.size()[-2] == self.image_size // 4:
                    frame1 = self.frame1(x)
                    frame_list.append(frame1)
                if x.size()[-1] == self.image_size // 2 and x.size()[-2] == self.image_size // 2:
                    frame2 = self.frame2(x)
                    frame_list.append(frame2)

        x = decoder[-1](x)
        if self.multi_size:
            return x, frame_list
        else:
            return x

    def forward(self, inp):
        ## reflectance part 
        x1 = inp
        encoded1 = []
        for ind in range(len(self.encoder1)):
            x1 = self.encoder1[ind](x1)
            if self.act == 'relu':
                x1 = F.leaky_relu(x1, inplace=True)
            else:
                x1 = F.elu(x1, inplace=True)
            if self.detach:
                encoded1.append(x1.detach())
            else:
                encoded1.append(x1)
        if self.low_se:
            x1 = self.se_layer(x1)
        if self.multi_size:
            x1, frame_list = self.__decode(self.decoder_reflectance, encoded1, x1)
            return x1, frame_list
        else:
            x1 = self.__decode(self.decoder_reflectance, encoded1, x1)
            return x1

class SEComposer(nn.Module):

    def __init__(self, reflectance, shading, refl_multi_size, shad_multi_size):
        super(SEComposer, self).__init__()

        self.reflectance = reflectance
        self.shading = shading
        self.refl_multi_size = refl_multi_size
        self.shad_multi_size = shad_multi_size

    def forward(self, inp):
        if self.refl_multi_size:
            reflectance, reflectance_list = self.reflectance(inp)
        else:
            reflectance = self.reflectance(inp)
        if self.shad_multi_size:
            shading, shading_list = self.shading(inp)
        else:
            shading = self.shading(inp)
    
        if self.refl_multi_size and self.shad_multi_size:
            return reflectance, shading, reflectance_list, shading_list
        elif self.refl_multi_size:
            return reflectance, shading, reflectance_list
        elif self.shad_multi_size:
            return reflectance, shading, shading_list
        else:
            return reflectance, shading

class SEDecomposer(nn.Module):

    def __init__(self, channels=[3, 32, 64, 128, 256], kernel_size=3, padding=1, skip_se=False, se_improved=True, dilation_flag=False):
        super(SEDecomposer, self).__init__()

        stride_fn = lambda ind: 1 if ind==0 else 2
        if dilation_flag:
            self.dilation_channels = [1, 8, 4, 2]
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder1 = build_encoder(channels, kernel_size, padding, stride_fn, dilation=self.dilation_channels if dilation_flag else None)
        self.encoder2 = build_encoder(channels, kernel_size, padding, stride_fn, dilation=self.dilation_channels if dilation_flag else None)
        self.skip_se = skip_se
        self.se_improved = se_improved
        if self.skip_se:
            sys.stdout.write( 'skip SELayer on\n' )
            self.se_skip_layers = []
            for c in channels[1:]:
                self.se_skip_layers.append(SELayerImproved(channel=c) if self.se_improved else SELayer(channel=c))
            self.se_skip_modulelist = nn.ModuleList(self.se_skip_layers)
        else:
            sys.stdout.write( 'skip SELayer off\n' )
        if self.se_improved:
            sys.stdout.write( 'SELayer improve on\n' )
        else:
            sys.stdout.write( 'SELayer improve off\n' )
        ## link encoder and decoder
        channels.append(channels[-1])
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Decomposer> Building Decoder' )
        
        # channels[0] = sum([channels[0] // (2**i) for i in range(stage + 1)])
        self.decoder_reflectance = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        self.decoder_shading = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)

        self.upsampler = nn.Upsample(scale_factor=2)

        self.se_layer_refl = SELayerImproved(channels[0]) if self.se_improved else SELayer(channels[0])
        self.se_layer_shad = SELayerImproved(channels[0]) if self.se_improved else SELayer(channels[0])

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            if self.skip_se:
                encoded[-(ind+1)] = self.se_skip_modulelist[-(ind+1)](encoded[-(ind+1)])
            x = torch.cat( (x, encoded[-(ind+1)]), 1)
            x = F.leaky_relu(x, inplace=True)

        x = decoder[-1](x)
        return x

    def forward(self, inp):
        ## reflectance part 
        x1 = inp
        encoded1 = []
        for ind in range(len(self.encoder1)):
            x1 = self.encoder1[ind](x1)
            x1 = F.leaky_relu(x1, inplace=True)
            encoded1.append(x1)
        x1 = self.se_layer_refl(x1)
        ## shading part 
        x2 = inp
        encoded2 = []
        for ind in range(len(self.encoder2)):
            x2 = self.encoder2[ind](x2)
            x2 = F.leaky_relu(x2, inplace=True)
            encoded2.append(x2)
        x2 = self.se_layer_shad(x2)
        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded1, x1)
        shading = self.__decode(self.decoder_shading, encoded2, x2)

        return reflectance, shading

def StrToBool(s):
    if s.lower() == 'false':
        return False
    else:
        return True

if __name__ == '__main__':
    inp = Variable(torch.randn(1,3,440,1024)).to('cuda')
    decomposer = SEDecomposerSingle(skip_se=True, low_se=True, multi_size=True, se_squeeze=True).to('cuda')
    out, f_list = decomposer.forward(inp)
    print(out.size())
    print([i.size() for i in f_list])