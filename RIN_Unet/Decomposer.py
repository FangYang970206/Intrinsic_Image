import sys, torch, torch.nn as nn, torch.nn.functional as F,torch.optim as optim
from torch.autograd import Variable
from .primitives import build_encoder, join, normalize


'''
Predicts reflectance, shape, and lighting conditions given an image

Reflectance and shape are 3-channel images of the 
same dimensionality as input (expects 256x256). 
Lights have dimensionality lights_dim. By default,
they are represented as [x, y, z, energy].
'''
class Decomposer(nn.Module):

    def __init__(self):
        super(Decomposer, self).__init__()

        #######################
        #### shape encoder #### 
        #######################
        ## there is a single shared convolutional encoder
        ## for all intrinsic images
        channels = [3, 16, 32, 64, 128]
        kernel_size = 3
        padding = 1
        ## stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder1 = build_encoder(channels, kernel_size, padding, stride_fn)
        self.encoder2 = build_encoder(channels, kernel_size, padding, stride_fn)
        #######################
        #### shape decoder #### 
        #######################
        ## link encoder and decoder
        channels.append( channels[-1] )
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Decomposer> Building Decoder' )
        ## separate reflectance and normals decoders.
        ## mult = 2 because the skip layer concatenates
        ## an encoder layer with the decoder layer,
        ## so the number of input channels in each layer is doubled.
        self.decoder_reflectance = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        self.decoder_shading = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        channels[-1] = 1
        self.upsampler = nn.Upsample(scale_factor=2)

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            x = join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x, inplace=True)

        x = decoder[-1](x)
        return x

    def forward(self, inp):
        ## shared encoder
        x1 = inp
        x2 = inp
        encoded1 = []
        encoded2 = []
        for ind in range(len(self.encoder1)):
            x1 = self.encoder1[ind](x1)
            x1 = F.leaky_relu(x1, inplace=True)
            encoded1.append(x1)
        for ind in range(len(self.encoder2)):
            x2 = self.encoder2[ind](x2)
            x2 = F.leaky_relu(x2, inplace=True)
            encoded2.append(x2)

        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded1, x1)
        shading = self.__decode(self.decoder_shading, encoded2, x2)

        return reflectance, shading


if __name__ == '__main__':
    inp = Variable(torch.randn(5,3,256,256))
    mask = Variable(torch.randn(5,3,256,256))
    decomposer = Decomposer()
    out = decomposer.forward(inp)
    print(decomposer)
    print([i.size() for i in out])



