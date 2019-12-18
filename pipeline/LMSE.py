import torch
import torch.nn as nn



class LMSE(nn.Module):
    '''
    
    LMSE loss : written for BOLD evaluation

    function __init__ :
    args defalut: slidingwindow_size = 10 
    this window will slide one half of sildingwindow_size in image

    function forword:
    args: input--estimated image
          target--ground truth
          image size need to follow [batch,channle,height,width]

    '''
    def __init__(self, slidingwindow_size=10):
        super(LMSE, self).__init__()
        self.k = slidingwindow_size
        self.loss = nn.MSELoss()
    
    def forward(self, input, target):
        assert input.size() == target.size()
        lmse_loss = torch.cuda.FloatTensor([0])
        _, _, row, column = input.size()
        count = 0
        for i in range(0, row-self.k, int(self.k/2)):
            for j in range(0, column-self.k, int(self.k/2)):
                lmse_loss += self.loss(input[:,:,i:i+self.k,j:j+self.k], 
                                        target[:,:,i:i+self.k,j:j+self.k]) 
                count += 1       
        return lmse_loss/count

if __name__ == '__main__':
    device = torch.device("cuda")
    a = torch.randn(5,4,28,28).to(device)
    b = torch.randn(5,4,28,28).to(device)
    lmse = LMSE().to(device)
    loss = lmse(a,b)
    print(loss)
    print(loss.item())