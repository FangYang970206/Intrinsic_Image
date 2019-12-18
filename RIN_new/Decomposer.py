import sys, torch, torch.nn as nn, torch.nn.functional as F,torch.optim as optim
from torch.autograd import Variable
if __name__ == "__main__":
    from primitives import build_encoder, join
else:
    from .primitives import build_encoder, join

from torch import nn


def conv(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    batch_norm = nn.BatchNorm2d(out_channels)
    layer = nn.Sequential(convolution, batch_norm)
    return layer

class Stage(nn.Module):
    def __init__(self, stage, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Stage, self).__init__()
        layers = []
        for i in range(1, stage + 1):
            layers.append(conv(in_channels * (2**i - 1), out_channels * (2**i - 1), kernel_size, stride, padding, dilation, groups, bias))
        self.stage = stage
        self.moduleList = nn.ModuleList(layers)
        self.act = nn.LeakyReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, x, encoded):
        if not isinstance(encoded, list):
            raise ValueError("encode should be list")
        if len(encoded) != self.stage:
            raise ValueError("the length of encode must be {}".format(self.stage))
        for i in range(len(self.moduleList)):
            x = self.moduleList[i](x)
            x = self.act(x)
            x = self.up(x)
            if self.stage - i != 1:
                x = torch.cat((encoded[self.stage - (2 + i)], x), 1)
        return x

class Decomposer(nn.Module):

    def __init__(self, channels=[3, 16, 32, 64, 128], kernel_size=3, padding=1, stage=3):
        super(Decomposer, self).__init__()

        # stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder1 = build_encoder(channels, kernel_size, padding, stride_fn)
        self.encoder2 = build_encoder(channels, kernel_size, padding, stride_fn)
        ## link encoder and decoder
        channels.append(channels[-1])
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Decomposer> Building Decoder' )
        
        channels[0] = sum([channels[0] // (2**i) for i in range(stage + 1)])
        self.decoder_reflectance = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        self.decoder_shading = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)

        self.upsampler = nn.Upsample(scale_factor=2)

        stage_channels = [channels[1] // (2**i) for i in range(stage + 1)]
        stride_fn = lambda ind: 2

        sys.stdout.write( '<Decomposer> Stage Downsample ' )
        self.stage_refl_downsample = build_encoder(stage_channels, kernel_size, padding, stride_fn)
        self.stage_shad_downsample = build_encoder(stage_channels, kernel_size, padding, stride_fn)

        sys.stdout.write( '<Decomposer> Stage Concat ' )
        self.stage_refl_concat = Stage(stage, stage_channels[-1], stage_channels[-1])
        self.stage_shad_concat = Stage(stage, stage_channels[-1], stage_channels[-1])

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
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
        s1 = x1
        encoded3 = []
        for ind in range(len(self.stage_refl_downsample)):
            s1 = self.stage_refl_downsample[ind](s1)
            s1 = F.leaky_relu(s1, inplace=True)
            encoded3.append(s1)
        s1 = self.stage_refl_concat(s1, encoded3)
        x1 = torch.cat( (s1, x1), 1)
        ## shading part 
        x2 = inp
        encoded2 = []
        for ind in range(len(self.encoder2)):
            x2 = self.encoder2[ind](x2)
            x2 = F.leaky_relu(x2, inplace=True)
            encoded2.append(x2)
        s2 = x2
        encoded4 = []
        for ind in range(len(self.stage_shad_downsample)):
            s2 = self.stage_shad_downsample[ind](s2)
            s2 = F.leaky_relu(s2, inplace=True)
            encoded4.append(s2)
        s2 = self.stage_shad_concat(s2, encoded4)
        x2 = torch.cat( (s2, x2), 1)

        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded1, x1)
        shading = self.__decode(self.decoder_shading, encoded2, x2)

        return reflectance, shading

if __name__ == '__main__':
    inp = Variable(torch.randn(5,3,256,256))
    decomposer = Decomposer(stage=1)
    out = decomposer.forward(inp)
    print([i.size() for i in out])