import sys, torch
import torch.nn as nn
import torch.nn.functional as F


class Composer(nn.Module):

    def __init__(self, reflection, shader):
        super(Composer, self).__init__()

        self.reflection = reflection
        self.shader = shader

    def forward(self, x):
        reflectance = self.reflection(x)
        shading = self.shader(x)
        shading_rep = shading.repeat(1,3,1,1)
        reconstruction = reflectance * shading_rep
        return reconstruction, reflectance, shading


if __name__ == '__main__':
    from Reflection import Reflection
    from Shader import Shader
    reflection = Reflection()
    shader = Shader()
    composer = Composer(reflection, shader).cuda()
    print(composer)
    inp = torch.randn(5, 3, 512, 512).cuda()
    out = composer.forward(inp)
    print([i.size() for i in out])






