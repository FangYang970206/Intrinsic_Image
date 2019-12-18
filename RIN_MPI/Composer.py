import sys, torch, torch.nn as nn, torch.nn.functional as F, pdb
from torch.autograd import Variable

class Composer(nn.Module):

    def __init__(self, decomposer, shader):
        super(Composer, self).__init__()

        self.decomposer = decomposer
        self.shader = shader

    def forward(self, inp):
        reflectance, shape, lights = self.decomposer(inp)
        # print(reflectance.size(), lights.size())
        shading = self.shader(shape, lights)
        # shading = shading.repeat(1,3,1,1)
        # print(shading.size())
        # print(reflectance.size())
        # print(shading.size())
        if self.shader.output_ch == 1:
            reconstruction = reflectance * shading.repeat(1,3,1,1)
            return reconstruction, reflectance, shading.repeat(1,3,1,1), shape
        else:
            reconstruction = reflectance * shading
            return reconstruction, reflectance, shading, shape
        # return reflectance, shading, shape 
        


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import models
    decomposer_path = '../logs/separated_decomp_0.01lr_0.1lights/model.t7'
    shader_path = '../logs/separated_shader_0.01/model.t7'
    decomposer = torch.load(decomposer_path)
    shader = torch.load(shader_path)
    composer = Composer(decomposer, shader).cuda()
    print(composer)
    # pdb.set_trace()
    inp = Variable(torch.randn(5,3,256,256).cuda())
    mask = Variable(torch.randn(5,3,256,256).cuda())

    out = composer.forward(inp, mask)

    print([i.size() for i in out])

    # pdb.set_trace()




