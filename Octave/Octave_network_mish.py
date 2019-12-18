import torch
import torch.nn as nn
if __name__ == "__main__":
    from OctaveConvMish import *
    from mish import Mish
else:
    from .OctaveConvMish import *
    from .mish import Mish


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=stride, bias=False, padding=0)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 norm_layer=None,First=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.first = First
        if self.first:
            self.ocb1 = FirstOctaveCBR(inplanes, planes, kernel_size=(1, 1),norm_layer=norm_layer,padding=0)
        else:
            self.ocb1 = OctaveCBR(inplanes, planes, kernel_size=(1,1),norm_layer=norm_layer,padding=0)

        self.ocb2 = OctaveCBR(planes, planes, kernel_size=(3,3), stride=stride, groups=groups, norm_layer=norm_layer)

        self.ocb3 = OctaveCB(planes, planes, kernel_size=(1,1), norm_layer=norm_layer,padding=0)
        self.relu = Mish()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.first:
            x_h_res, x_l_res = self.ocb1(x)
            x_h, x_l = self.ocb2((x_h_res, x_l_res))
        else:
            x_h_res, x_l_res = x
            x_h, x_l = self.ocb1((x_h_res,x_l_res))
            x_h, x_l = self.ocb2((x_h, x_l))

        # print("x_l_res: ", x_l_res.size())
        # print("x_h_res: ", x_h_res.size())
        # print("x_h: ", x_h.size())
        # print("x_l: ", x_l.size())

        x_h, x_l = self.ocb3((x_h, x_l))

        if self.downsample is not None:
            x_h_res, x_l_res = self.downsample((x_h_res,x_l_res))

        x_h += x_h_res
        x_l += x_l_res

        x_h = self.relu(x_h)
        x_l = self.relu(x_l)

        return x_h, x_l

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = Mish()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class OctaveIID(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, norm_layer=None):
        super(OctaveIID, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 32
        self.groups = groups
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = Mish()
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, First=True)
        self.octaveCBR1 = OctaveCBR(64, 128)
        self.layer2 = self._make_layer(block, 128, layers[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 128, layers[2], norm_layer=norm_layer)
        self.octaveCBR2 = OctaveCBR(128, 64)
        self.layer4 = self._make_layer(block, 64, layers[3], norm_layer=norm_layer)
        self.basic_block_h = BasicBlock(32, 32)
        self.basic_block_l = BasicBlock(32, 32)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.conv2_h = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_l = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.inplanes,out_channels=planes, kernel_size=(1,1), stride=stride, padding=0)
            )

        layers = []
        layers.append(block(self.inplanes if First else planes, planes, stride, downsample, self.groups,
                        norm_layer, First))
        # self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, planes, groups=self.groups,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(1, x.size())
        x_h, x_l = self.layer1(x)
        x_h, x_l = self.octaveCBR1((x_h, x_l))
        # print(2, x_h.size())
        # print(2, x_l.size())
        x_h, x_l = self.layer2((x_h,x_l))
        # print(3, x_h.size())
        # print(3, x_l.size())
        x_h, x_l = self.layer3((x_h,x_l))
        x_h, x_l = self.octaveCBR2((x_h, x_l))
        # print(4, x_h.size())
        # print(4, x_l.size())
        x_h, x_l = self.layer4((x_h,x_l))
        # print(5, x_h.size())
        # print(5, x_l.size())
        x_h = self.basic_block_h(x_h)
        x_l = self.Upsample(x_l)
        x_l = self.basic_block_l(x_l)
        x_h = self.Upsample(x_h)
        x_l = self.Upsample(x_l)
        x_h = self.conv2_h(x_h)
        x_l = self.conv2_l(x_l)
        x_l = self.sigmoid(x_l)
        x_h = self.sigmoid(x_h)
        return x_h, x_l
        
def get_mish_model_two_output(layers=[2,2,2,2]):
    model = OctaveIID(Bottleneck, layers)
    return model
    

if __name__ == '__main__':
    iid = OctaveIID(Bottleneck, [3,5,5,3]).to('cuda')
    # print(iid)
    t = torch.randn(1, 3, 436, 1024).to('cuda')
    out = iid(t)
    print([i.size() for i in out])