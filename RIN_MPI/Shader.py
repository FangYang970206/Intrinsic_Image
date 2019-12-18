import torch, torch.nn as nn, torch.nn.functional as F
import sys
from torch.autograd import Variable
from .primitives import build_encoder, join, normalize

'''
Predicts shading image given shape and lighting conditions.

Shading image is of the same dimensionality as shape 
(expects 256x256). Lights have dimensionality lights_dim. 
By default, they are represented as [x, y, z, energy].
'''
class Shader(nn.Module):

    def __init__(self, lights_dim=4, expand_dim=32, output_ch=1): #change the expand_dim(8-->32) ,delete two layers[32,256]
        super(Shader, self).__init__()
        #### shape encoder
        channels = [3, 16, 64, 128, 256]
        kernel_size = 3
        padding = 1
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Shader> Building Encoder' )
        self.encoder = build_encoder(channels, kernel_size, padding, stride_fn)
        # self.encoder = nn.ModuleList( self.encoder )
        self.remove_padding = nn.ReplicationPad2d((0,0,-6,-6))
        #### shape decoder
        ## link encoder and decoder
        channels.append( channels[-1] )
        ## single channel shading output
        self.output_ch = output_ch
        channels[0] = output_ch
        ## add a channel for the lighting
        channels[-1] += 1
        ## reverse order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Shader> Building Decoder ' )
        self.decoder = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        # self.decoder = nn.ModuleList( self.decoder )
        self.upsampler = nn.Upsample(scale_factor=2)

        #### lights encoder
        ## same dimensionality as encoded shape
        self.expand_dim = expand_dim
        self.lights_fc = nn.Linear(lights_dim, 56 * 128)

    ## x is normals < batch x 3 x 256 x 256 >
    ## lights is < batch x 4 > [x, y, z, energy] 
    def forward(self, x, lights):
        ## forward shape
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        ## forward lights
        lights = self.lights_fc(lights)
        lights = lights.view(-1, 1, 56, 128)
        
        ## concatenate shape and lights representations
        x = torch.cat( (encoded[-1], lights), 1 )

        ## decode concatenated representation
        ## with skip layers from the encoder
        for ind in range(len(self.decoder)-1):
            x = self.decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            x = join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x)

        x = self.decoder[-1](x)

        return self.remove_padding(x)


if __name__ == '__main__':
    shape = Variable(torch.randn(1,3,448,1024))
    lights = Variable(torch.randn(1,4))
    shader = Shader()
    out = shader.forward(shape, lights)
    print (out.size())



