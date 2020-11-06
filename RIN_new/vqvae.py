import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    def __init__(self, 
                in_channel, 
                norm_layer,
        ):
        super().__init__()
        self.conv1 = conv1x1(in_channel, in_channel // 4)
        self.bn1 = norm_layer(in_channel // 4)
        self.conv2 = conv3x3(in_channel // 4, in_channel // 4)
        self.bn2 = norm_layer(in_channel // 4)
        self.conv3 = conv1x1(in_channel // 4, in_channel)
        self.bn3 = norm_layer(in_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.act(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, norm_layer, use_inception=False):
        super().__init__()

        blocks = [
            nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
            norm_layer(channel),
            nn.LeakyReLU(inplace=True),
        ]

        if use_inception:
            blocks.append(Inception(channel, channel // 4, channel // 4, channel // 4, channel // 4, channel // 4, channel // 4))
        else:
            blocks.append(ResBlock(channel, norm_layer))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, norm_layer, is_last=False, use_tanh=False):
        super().__init__()

        if is_last:
            blocks = [Inception(in_channel, channel // 4, channel // 4, channel // 4, channel // 4, channel // 4, channel // 4)]
        else:
            blocks = [
                nn.Conv2d(in_channel, channel, 3, stride=1, padding=1),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(inplace=True),
            ]

        blocks.append(nn.Upsample(scale_factor=2))
        blocks.append(ResBlock(channel, norm_layer))        
        blocks.append(nn.Conv2d(channel, out_channel, 3, stride=1, padding=1))
        if not is_last:
            blocks.append(norm_layer(out_channel))
            blocks.append(nn.LeakyReLU(inplace=True))
        else:
            if use_tanh:
                blocks.append(nn.Tanh())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        vq_flag=True,
        norm_layer=None,
        init_weights=None,
        use_tanh=False,
        use_inception=False,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.enc_b = Encoder(in_channel, channel, norm_layer, use_inception=use_inception)
        self.enc_m = Encoder(channel, channel, norm_layer, use_inception=use_inception)
        self.enc_t = Encoder(channel, channel, norm_layer, use_inception=use_inception)

        self.res_block1 = ResBlock(channel * 2, norm_layer)
        self.res_block2 = ResBlock(channel * 2, norm_layer)
        
        self.dec_t = Decoder(channel, channel, channel, norm_layer)
        self.dec_m = Decoder(channel*2, channel, channel, norm_layer)
        self.dec_b = Decoder(channel*5, in_channel, channel, norm_layer, is_last=True, use_tanh=use_tanh)
        
        self.up = nn.Upsample(scale_factor=2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        enc_b = self.enc_b(input)
        enc_m = self.enc_m(enc_b)
        enc_t = self.enc_t(enc_m)

        dec_t = self.dec_t(enc_t)
        cat_mt = torch.cat([enc_m, dec_t], 1)
        cat_res_mt = self.res_block1(cat_mt)
        
        up1 = self.up(enc_t)
        scale1 = torch.cat([up1, cat_res_mt], 1)

        dec_m = self.dec_m(cat_mt)
        cat_bm = torch.cat([dec_m, enc_b], 1)
        cat_res_bm = self.res_block2(cat_bm)

        up2 = self.up(scale1)
        scale2 = torch.cat([up2, cat_res_bm], 1)

        out = self.dec_b(scale2)
        return out

if __name__ == "__main__":
    vae = VQVAE(use_tanh=True)
    t = torch.randn(2, 3, 256, 256)
    out = vae(t)
    print(out.size())
    vae = VQVAE(init_weights=True, use_tanh=True, use_inception=True)
    t = torch.randn(2, 3, 256, 256)
    out = vae(t)
    print(out.size())
    vae = VQVAE(init_weights=True, use_tanh=True)
    t = torch.randn(2, 3, 256, 256)
    out = vae(t)
    print(out.size())
    # for t in out:
    #     print(t.size())

# class VQVAE(nn.Module):
#     def __init__(
#         self,
#         in_channel=3,
#         channel=128,
#         n_res_block=2,
#         n_res_channel=32,
#         embed_dim=64,
#         n_embed=512,
#         decay=0.99,
#         vq_flag=True,
#     ):
#         super().__init__()

#         self.vq_flag = vq_flag
#         self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
#         self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
#         self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
#         if self.vq_flag:
#             self.quantize_t = Quantize(embed_dim, n_embed)
#         self.dec_t = Decoder(
#             embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
#         )
#         self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
#         if self.vq_flag:
#             self.quantize_b = Quantize(embed_dim, n_embed)
#         self.upsample_t = nn.ConvTranspose2d(
#             embed_dim, embed_dim, 4, stride=2, padding=1
#         )
#         self.dec = Decoder(
#             embed_dim + embed_dim,
#             in_channel,
#             channel,
#             n_res_block,
#             n_res_channel,
#             stride=4,
#         )

#     def forward(self, input):
#         if self.vq_flag:
#             quant_t, quant_b, diff, _, _ = self.encode(input)
#         else:
#             quant_t, quant_b = self.encode(input)
#         dec = self.decode(quant_t, quant_b)
#         if self.vq_flag:
#             return dec, diff
#         else:
#             return dec

#     def encode(self, input):
#         enc_b = self.enc_b(input)
#         enc_t = self.enc_t(enc_b)

#         if self.vq_flag:
#             quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
#             quant_t, diff_t, id_t = self.quantize_t(quant_t)
#             quant_t = quant_t.permute(0, 3, 1, 2)
#             diff_t = diff_t.unsqueeze(0)
#         else:
#             quant_t = self.quantize_conv_t(enc_t)

#         dec_t = self.dec_t(quant_t)
#         enc_b = torch.cat([dec_t, enc_b], 1)

#         if self.vq_flag:
#             quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
#             quant_b, diff_b, id_b = self.quantize_b(quant_b)
#             quant_b = quant_b.permute(0, 3, 1, 2)
#             diff_b = diff_b.unsqueeze(0)
#             return quant_t, quant_b, diff_t + diff_b, id_t, id_b
#         else:
#             quant_b = self.quantize_conv_b(enc_b)
#             return quant_t, quant_b
        

#     def decode(self, quant_t, quant_b):
#         upsample_t = self.upsample_t(quant_t)
#         quant = torch.cat([upsample_t, quant_b], 1)
#         dec = self.dec(quant)

#         return dec

#     def decode_code(self, code_t, code_b):
#         quant_t = self.quantize_t.embed_code(code_t)
#         quant_t = quant_t.permute(0, 3, 1, 2)
#         quant_b = self.quantize_b.embed_code(code_b)
#         quant_b = quant_b.permute(0, 3, 1, 2)

#         dec = self.decode(quant_t, quant_b)

#         return dec