import torch
import torch.nn as nn
import torchvision.models as models



class VGG_Encoder(nn.Module):
    def __init__(self, device):
        super(VGG_Encoder, self).__init__()
        self.vgg = models.vgg11().features
        self.vgg = self.vgg.to(device)
        state_dict = torch.load('RIN_pipeline/pretrained_model/vgg11-bbd30ac9.pth')
        # state_dict = torch.load('pretrained_model/vgg11-bbd30ac9.pth')
        self.index = [0, 3, 6, 8, 11, 13, 16, 18]
        
        for index in self.index:
            self.vgg[index].weight.data.copy_(state_dict['features.'+str(index)+'.weight'])
            self.vgg[index].bias.data.copy_(state_dict['features.'+str(index)+'.bias'])

        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self, fake, real):
        calc_index = [0, 3, 6, 8, 11, 13, 16, 18]
        if fake.size()[1] == 1:
            fake = fake.repeat(1, 3, 1, 1)
        if real.size()[1] == 1:
            real = real.repeat(1, 3, 1, 1)
        
        content_loss = 0
        for idx, sub_module in enumerate(self.vgg):
            fake = sub_module(fake)
            real = sub_module(real)
            if idx in calc_index:
                content_loss += ((fake-real)**2).mean()
        return content_loss/len(calc_index)
    # def forward(self, inp):
    #     return self.vgg(inp)


if __name__ == "__main__":
    from modelsummary import summary
    model = VGG_Encoder()
    input = torch.randn(4, 3, 256, 256).to('cuda')
    summary(model, input)
    output = model.forward(input)
    print(output.size())