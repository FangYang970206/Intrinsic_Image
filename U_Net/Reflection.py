import torch.nn as nn
import torch.nn.functional as F
import sys, torch

from .primitives import build_encoder, join


'''
Predicts reflectance given an image
'''
class Reflection(nn.Module):

    def __init__(self):
        super(Reflection, self).__init__()
        ## there is a single shared convolutional encoder
        ## for all intrinsic images
        channels = [3, 16, 64, 128, 256]
        kernel_size = 3
        padding = 1
        ## stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Reflectance> Building Encoder' )
        self.encoder = build_encoder(channels, kernel_size, padding, stride_fn)
        ## link encoder and decoder
        channels.append( channels[-1] )
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Reflectance> Building Decoder' )
        ## separate reflectance and normals decoders.
        ## mult = 2 because the skip layer concatenates
        ## an encoder layer with the decoder layer,
        ## so the number of input channels in each layer is doubled.
        self.decoder_reflectance = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = F.interpolate(x, scale_factor=2)
            x = join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x)

        x = decoder[-1](x)
        return x

    def forward(self, x):
        ## shared encoder
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded, x)
        
        return reflectance


if __name__ == '__main__':
    inp = torch.randn(5,3,512,512)
    Reflection = Reflection()
    out = Reflection.forward(inp)
    print([i.size() for i in out])



