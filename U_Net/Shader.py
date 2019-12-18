import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .primitives import build_encoder, join

'''
Predicts shading given an image
'''
class Shader(nn.Module):

    def __init__(self): #change the expand_dim(8-->32) ,delete two layers[32,256]
        super(Shader, self).__init__()
        #### shape encoder
        channels = [3, 16, 64, 128, 256]
        kernel_size = 3
        padding = 1
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Shader> Building Encoder' )
        self.encoder = build_encoder(channels, kernel_size, padding, stride_fn)
        # self.encoder = nn.ModuleList( self.encoder )

        #### shape decoder
        ## link encoder and decoder
        channels.append( channels[-1] )
        ## single channel shading output
        channels[0] = 1
        ## add a channel for the lighting
        # channels[-1] += 1
        ## reverse order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Shader> Building Decoder ' )
        self.decoder = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)

    def forward(self, x):
        ## forward shape
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x, inplace=True)
            encoded.append(x)

        for ind in range(len(self.decoder)-1):
            x = self.decoder[ind](x)
            if ind != 0:
                x = F.interpolate(x, scale_factor=2)
            x = join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x, inplace=True)

        x = self.decoder[-1](x)

        return x


if __name__ == '__main__':
    shape = torch.randn(5, 3, 512, 512)
    shader = Shader()
    out = shader.forward(shape)
    print(out.size())



