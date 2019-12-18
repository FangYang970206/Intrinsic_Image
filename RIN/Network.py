import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=True):
        super(conv_block, self).__init__()
        if downsample:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out), nn.LeakyReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch_out), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=True, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        # print(ch_in, ch_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in), nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in), nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in), nn.LeakyReLU(inplace=True))
        if downsample:
            self.conv4 = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch_out), nn.LeakyReLU(inplace=True))
        else:
            self.conv4 = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch_out), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv1(x)
                x1 = self.conv2(x + x1)
            else:
                x2 = self.conv3(x1)
                x2 = self.conv4(x1+x2)
        return x2


class RCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=True, t=2):
        super(RCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_in, ch_out, downsample=downsample, t=t))
        if downsample:
            self.Conv_3x3 = nn.Conv2d(
                ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        else:
            self.Conv_3x3 = nn.Conv2d(
                ch_in, ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.Conv_3x3(x)
        x2 = self.RCNN(x)
        return x1 + x2


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=True, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_in, ch_in, downsample=False, t=t),
            Recurrent_block(ch_in, ch_out, downsample=downsample, t=t))
        if downsample:
            self.Conv_3x3 = nn.Conv2d(
                ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        else:
            self.Conv_3x3 = nn.Conv2d(
                ch_in, ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.Conv_3x3(x)
        x2 = self.RCNN(x)
        return x1 + x2


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def build_encoder(channels,
                  rin_only=False,
                  use_rcnn_block=False,
                  use_rrcnn_block=False,
                  use_attention=False,
                  downsample=False,
                  mult=1):
    layers = []
    # sys.stdout.write( '    %3d' % channels[0])
    for ind in range(len(channels) - 1):
        m = 1 if ind == 0 else mult
        in_channels = channels[ind] * m
        out_channels = channels[ind + 1]
        # print('{}--->{} \n'.format(in_channels, out_channels))
        # stride = stride_fn(ind)
        if use_attention:
            if ind < len(channels) - 1:
                block = Attention_block(F_g=in_channels, F_l=in_channels, F_int=out_channels)
            else:
                return nn.ModuleList(layers)
        else:
            if ind < len(channels) - 2:
                if rin_only:
                    if ind==0:
                        block = conv_block(ch_in=in_channels,
                                           ch_out=out_channels, 
                                           downsample=False)
                    else:
                        block = conv_block(ch_in=in_channels,
                                           ch_out=out_channels, 
                                           downsample=downsample)
                elif use_rcnn_block:
                    if ind==0:
                        block = RCNN_block(ch_in=in_channels, 
                                           ch_out=out_channels, 
                                           downsample=False)
                    else:
                        block = RCNN_block(ch_in=in_channels, 
                                           ch_out=out_channels, 
                                           downsample=downsample)
                elif use_rrcnn_block:
                    if ind==0:
                        block = RRCNN_block(ch_in=in_channels, 
                                            ch_out=out_channels, 
                                            downsample=False)
                    else:
                        block = RRCNN_block(ch_in=in_channels, 
                                            ch_out=out_channels, 
                                            downsample=downsample)
            else:
                if downsample:
                    block = nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1)
                else:
                    block = nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1)

        layers.append(block)
    return nn.ModuleList(layers)


def normalize(normals):
    magnitude = torch.pow(normals, 2).sum(1,keepdim=True)
    magnitude = magnitude.sqrt().repeat(1,3,1,1) 
    normed = normals / (magnitude + 1e-6)
    return normed


class Decomposer(nn.Module):
    def __init__(self,
                 lights_dim=4,
                 rin_only=False,
                 use_rcnn_block=False,
                 use_rrcnn_block=False,
                 use_attention=False):
        super(Decomposer, self).__init__()
        
        self.rin_only = rin_only
        self.use_rcnn_block = use_rcnn_block
        self.use_rrcnn_block = use_rrcnn_block
        self.use_attention = use_attention
        
        refl_channels = [3, 16, 32, 64, 128, 128]
        shap_channels = [3, 16, 32, 64, 128, 128]
        lights_channels = [128, 64, 32]
        
        self.encoder1 = build_encoder(refl_channels,
                                      self.rin_only,
                                      self.use_rcnn_block,
                                      self.use_rrcnn_block,
                                      downsample=True)

        self.encoder2 = build_encoder(shap_channels,
                                      self.rin_only,
                                      self.use_rcnn_block,
                                      self.use_rrcnn_block,
                                      downsample=True)
        
        refl_channels.append(refl_channels[-1])
        shap_channels.append(shap_channels[-1])
        refl_channels = list(reversed(refl_channels))
        shap_channels = list(reversed(shap_channels))

        self.decoder_reflectance = build_encoder(refl_channels,
                                                 self.rin_only,
                                                 self.use_rcnn_block,
                                                 self.use_rrcnn_block,
                                                 mult=2)

        self.decoder_normals = build_encoder(shap_channels,
                                             self.rin_only,
                                             self.use_rcnn_block,
                                             self.use_rrcnn_block,
                                             mult=2)

        self.decoder_lights = build_encoder(lights_channels, 
                                            self.rin_only, 
                                            self.use_rcnn_block, 
                                            self.use_rrcnn_block,
                                            downsample=True)
        
        if use_attention:
            self.attention1 = build_encoder(refl_channels[1:],
                                           use_attention=self.use_attention)
            # print("att1", len(self.attention1))
            self.attention2 = build_encoder(shap_channels[1:],
                                           use_attention=self.use_attention)
            # print("att2", len(self.attention2))

        self.lights_fc1 = nn.Linear(lights_channels[-1] * (2**6), 32)
        self.lights_fc2 = nn.Linear(32, lights_dim)
        self.upsampler = nn.Upsample(scale_factor=2)

    def __decode(self, decoder, encoded, inp, attention=None):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            if self.use_attention:
                # print(ind)
                # print(x.size())
                # print(encoded[-(ind+1)].size())
                encoder_x = attention[ind](g=x, x=encoded[-(ind+1)])
                x = torch.cat((x, encoder_x), 1)
            else:
                x = torch.cat((x, encoded[-(ind+1)]), 1)
            # x = F.leaky_relu(x)

        x = decoder[-1](x)
        return x

    def forward(self, inp):
        x1 = inp
        x2 = inp
        encoded1 = []
        encoded2 = []
        for ind in range(len(self.encoder1)):
            x1 = self.encoder1[ind](x1)
            # x1 = F.leaky_relu(x1)
            encoded1.append(x1)
        for ind in range(len(self.encoder2)):
            x2 = self.encoder2[ind](x2)
            # x2 = F.leaky_relu(x2)
            encoded2.append(x2)
        ## decode lights
        lights = x2
        for ind in range(len(self.decoder_lights)):
            lights = self.decoder_lights[ind](lights)
            # lights = F.leaky_relu(lights)
        # print(lights.size())
        lights = lights.view(lights.size(0), -1)
        lights = F.leaky_relu(self.lights_fc1(lights))
        lights = self.lights_fc2(lights)

        ## separate decoders
        if self.use_attention:
            reflectance = self.__decode(self.decoder_reflectance, encoded1, x1, self.attention1)
            normals = self.__decode(self.decoder_normals, encoded2, x2, self.attention2)
        else:
            reflectance = self.__decode(self.decoder_reflectance, encoded1, x1)
            normals = self.__decode(self.decoder_normals, encoded2, x2)

        rg = torch.clamp(normals[:, :-1, :, :], -1, 1)
        b = torch.clamp(normals[:, -1, :, :].unsqueeze(1), 0, 1)
        clamped = torch.cat((rg, b), 1)
        normed = normalize(clamped)

        return reflectance, normed, lights


class Shader(nn.Module):
    def __init__(self,
                 lights_dim=4,
                 expand_dim=32,
                 rin_only=False,
                 use_rcnn_block=False,
                 use_rrcnn_block=False,
                 use_attention=False):
        super(Shader, self).__init__()
        
        self.rin_only = rin_only
        self.use_rcnn_block = use_rcnn_block
        self.use_rrcnn_block = use_rrcnn_block
        self.use_attention = use_attention

        channels = [3, 16, 64, 128, 256]

        self.encoder = build_encoder(channels,
                                     self.rin_only,
                                     self.use_rcnn_block,
                                     self.use_rrcnn_block,
                                     downsample=True)

        channels.append( channels[-1])
        channels[0] = 1
        channels[-1] += 1
        channels = list(reversed(channels))

        if use_attention:
            self.attention = build_encoder(channels[1:],
                                           use_attention=self.use_attention)
        
        self.decoder = build_encoder(channels, 
                                     self.rin_only,
                                     self.use_rcnn_block,
                                     self.use_rrcnn_block,
                                     mult=2)

        self.upsampler = nn.Upsample(scale_factor=2)
        
        self.expand_dim = expand_dim
        self.lights_fc = nn.Linear(lights_dim, expand_dim * expand_dim)
    
    def forward(self, x, lights):
        ## forward shape
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            # x = F.leaky_relu(x)
            encoded.append(x)

        ## forward lights
        lights = self.lights_fc(lights)
        lights = lights.view(-1, 1, self.expand_dim, self.expand_dim)
        
        ## concatenate shape and lights representations
        x = torch.cat( (encoded[-1], lights), 1 )

        ## decode concatenated representation
        ## with skip layers from the encoder
        for ind in range(len(self.decoder)-1):
            x = self.decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            if self.use_attention:
                # print(ind)
                # print(x.size())
                # print(encoded[-(ind+1)].size())
                encoder_x = self.attention[ind](g=x, x=encoded[-(ind+1)])
                x = torch.cat((x, encoder_x), 1)
            else:
                x = torch.cat((x, encoded[-(ind+1)]), 1)
            # x = F.leaky_relu(x)

        x = self.decoder[-1](x)

        return x 


class Composer(nn.Module):

    def __init__(self, decomposer, shader):
        super(Composer, self).__init__()

        self.decomposer = decomposer
        self.shader = shader

    def forward(self, inp):
        reflectance, shape, lights = self.decomposer(inp)
        shading = self.shader(shape, lights)
        shading_rep = shading.repeat(1, 3, 1, 1)
        reconstruction = reflectance * shading_rep
        return reconstruction, reflectance, shading, shape


if __name__ == "__main__":
    from modelsummary import summary
    shader = Shader(use_rrcnn_block=True, use_attention=True).to('cuda')
    lights = torch.randn(4, 4).to('cuda')
    shape = torch.randn(4, 3, 256, 256).to('cuda')
    datas = [shape, lights]
    summary(shader, *datas)
    x = shader(shape, lights)
    print(x.size())
    # decomposer = Decomposer(use_rcnn_block=True, use_attention=True).to('cuda')
    # inp = torch.randn(4, 3, 256, 256).to('cuda')
    # summary(decomposer, inp, show_hierarchical=True)
    # y = decomposer.forward(inp)
    # print(y[0].size())
    # print(y[1].size())
    # print(y[2].size())
    