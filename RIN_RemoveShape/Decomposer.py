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

    def __init__(self, lights_dim = 4):
        super(Decomposer, self).__init__()

        #######################
        #### shape encoder #### 
        #######################
        ## there is a single shared convolutional encoder
        ## for all intrinsic images
        channels = [3, 16, 64, 128, 128]
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
        # self.decoder_normals = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        channels[-1] = 1
        #self.decoder_depth = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        self.upsampler = nn.Upsample(scale_factor=2)

        #### lights encoder
        # lights_channels = [256, 128, 64]
        lights_channels = [128, 64, 32]
        stride_fn = lambda ind: 2
        sys.stdout.write( '<Decomposer> Lights Encoder  ' )
        self.decoder_lights = build_encoder(lights_channels, kernel_size, padding, stride_fn)
        lights_encoded_dim = 2
 
        self.lights_fc1 = nn.Linear(lights_channels[-1] * (lights_encoded_dim ** 6), 32) #change(**2--->**6)
        self.lights_fc2 = nn.Linear(32, lights_dim)

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            x = join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x)

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
            x1 = F.leaky_relu(x1)
            encoded1.append(x1)
        for ind in range(len(self.encoder2)):
            x2 = self.encoder2[ind](x2)
            x2 = F.leaky_relu(x2)
            encoded2.append(x2)
        ## decode lights
        lights = x2
        for ind in range(len(self.decoder_lights)):
            lights = self.decoder_lights[ind](lights)
            lights = F.leaky_relu(lights)
        lights = lights.view(lights.size(0), -1)
        lights = F.leaky_relu( self.lights_fc1(lights) )
        lights = self.lights_fc2(lights)

        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded1, x1)
        # normals = self.__decode(self.decoder_normals, encoded2, x2)

        # rg = torch.clamp(normals[:,:-1,:,:], -1, 1)
        # b = torch.clamp(normals[:,-1,:,:].unsqueeze(1), 0, 1)
        # clamped = torch.cat((rg, b), 1)
        # normed = normalize(clamped)

        # return reflectance, normed, lights
        return reflectance, lights, encoded2


if __name__ == '__main__':
    inp = Variable(torch.randn(5,3,256,256))
    mask = Variable(torch.randn(5,3,256,256))
    decomposer = Decomposer()
    out = decomposer.forward(inp)
    # print(decomposer)
    print([i.size() for i in out])



