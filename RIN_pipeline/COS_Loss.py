import torch
import torch.nn as nn


class COS_Loss(nn.Module):
    '''
    
    COS loss : written for BOLD 

    function __init__ :
    args defalut: slidingwindow_size = 3 
    this window will slide one half of sildingwindow_size in image

    function forword:
    args: input--estimated image
          target--ground truth
          image size need to follow [batch,channle,height,width]

    '''
    def __init__(self, slidingwindow_size=128):
        super(COS_Loss, self).__init__()
        self.k = slidingwindow_size
    
    def forward(self, input, target):
        # assert input.size() == target.size()
        cos_loss = torch.cuda.FloatTensor([0])
        _, _, row, column = input.size()
        count = 0
        for i in range(0, row-self.k, self.k):
            for j in range(0, column-self.k, self.k):
                inp_region = input[:,:,i:i+self.k,j:j+self.k]
                tar_region = target[:,:,i:i+self.k,j:j+self.k]
                cos_value = (inp_region*tar_region).sum()/(torch.sqrt((inp_region**2).sum())*torch.sqrt((tar_region**2).sum()))
                cos_loss = (cos_value - 1)**2
                count += 1       
        return cos_loss/count

if __name__ == '__main__':
    device = torch.device("cuda")
    a = torch.ones(5,4,28,28).to(device)
    b = torch.ones(5,4,28,28).to(device)
    lmse = COS_Loss().to(device)
    loss = lmse(a,b)
    print(loss)
    print(loss.item())