import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch import nn


def conv(in_channels, out_channels, kernel_size, stride, padding):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channels)
    layer = nn.Sequential(convolution, batch_norm)
    return layer

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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


class SESingleGenerator(nn.Module):

    def __init__(self, channels=[3, 32, 64, 128, 256], kernel_size=3, padding=1, skip_se=False, se_improved=False, multi_size=False, image_size=320):
        super(SESingleGenerator, self).__init__()

        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder1 = build_encoder(channels, kernel_size, padding, stride_fn)
        self.skip_se = skip_se
        self.se_improved = se_improved
        self.multi_size = multi_size
        self.image_size = image_size
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
        
        self.decoder_reflectance = build_encoder(channels, kernel_size, padding, stride_fn, mult=2)

        self.upsampler = nn.Upsample(scale_factor=2)

        self.se_layer = SELayerImproved(channels[0]) if self.se_improved else SELayer(channels[0])
        if self.multi_size:
            self.frame1 = nn.Conv2d(256, 3, 3, 1, 1)
            self.frame2 = nn.Conv2d(128, 3, 3, 1, 1)
            self.relu = nn.ReLU()

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
            x = F.leaky_relu(x, inplace=True)
            if self.multi_size:
                if x.size()[-1] == self.image_size // 4 and x.size()[-2] == self.image_size // 4:
                    frame1 = self.frame1(x)
                    frame1 = self.relu(frame1)
                    frame_list.append(frame1)
                if x.size()[-1] == self.image_size // 2 and x.size()[-2] == self.image_size // 2:
                    frame2 = self.frame2(x)
                    frame2 = self.relu(frame2)
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
            x1 = F.leaky_relu(x1, inplace=True)
            encoded1.append(x1)
        x1 = self.se_layer(x1)
        if self.multi_size:
            x1, frame_list = self.__decode(self.decoder_reflectance, encoded1, x1)
            return x1, frame_list
        else:
            x1 = self.__decode(self.decoder_reflectance, encoded1, x1)
            return x1

class SEComposerGenerater(nn.Module):

    def __init__(self, reflectance, shading, refl_multi_size, shad_multi_size):
        super(SEComposerGenerater, self).__init__()

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

class SEUG_Generater(nn.Module):

    def __init__(self, channels=[3, 32, 64, 128, 256], kernel_size=3, padding=1, skip_se=False, se_improved=False):
        super(SEUG_Generater, self).__init__()

        # stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder1 = build_encoder(channels, kernel_size, padding, stride_fn)
        self.encoder2 = build_encoder(channels, kernel_size, padding, stride_fn)
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

class SEUG_Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=5):
        super(SEUG_Discriminator, self).__init__()
        self.conv1_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(True))
        self.conv2_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(True))
        self.conv3_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(True))
        self.conv4_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(True))
        # self.se_layer = SELayer(ndf * 8)
        # for i in range(1, n_layers - 2): #第二，三层下采样，尺寸再缩4倍(32)，通道数为256
        #     mult = 2 ** (i - 1)
        #     model += [nn.utils.spectral_norm(
        #               nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=1, bias=True)),
        #               nn.LeakyReLU(True)]

        # mult = 2 ** (n_layers - 2 - 1)
        # model += [nn.utils.spectral_norm(
        #           nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=1, bias=True)),
        #           nn.LeakyReLU(True)]

        # Class Activation Map， 与生成器得类别激活图类似
        # mult = 2 ** (n_layers - 2)
        # self.se_layer = SELayer(ndf * 8)
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, bias=False))

        # self.model = nn.Sequential(*model)

    def forward(self, input):
        x1 = self.conv1_block(input)
        x2 = self.conv2_block(x1)
        x3 = self.conv3_block(x2)
        x4 = self.conv4_block(x3)
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        y4 = self.conv4(x4)
        # out = self.conv(x) #输出大小是32x32，其他与生成器类似
        return y1, y2, y3, y4

class SEUG_Discriminator_new(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=5):
        super(SEUG_Discriminator_new, self).__init__()
        self.conv1_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(0.1, True))
        self.conv2_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(0.1, True))
        self.conv3_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(0.1, True))
        self.conv4_block = nn.Sequential(nn.utils.spectral_norm(
                                         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True)),
                                         nn.LeakyReLU(0.1, True))
        self.conv = nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input):
        x1 = self.conv1_block(input)
        x2 = self.conv2_block(x1)
        x3 = self.conv3_block(x2)
        x4 = self.conv4_block(x3)
        y = self.conv(x4)
        return y


if __name__ == '__main__':
    inp = torch.randn(1,3,32,32).to('cuda')
    SEUG_D = SEUG_Discriminator_new().to('cuda')
    out = SEUG_D.forward(inp)
    print(out.size())