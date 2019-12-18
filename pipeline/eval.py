import torch 


def corr2(A,B):
    """
    calculate 2D-torchtensor correlation coefficient
    """
    assert A.size() == B.size()
    A_mean, B_mean = A.mean(), B.mean()
    cov = ((A-A_mean)*(B-B_mean)).sum().sum()
    A_var = ((A-A_mean)*(A-A_mean)).sum().sum()
    B_var = ((B-B_mean)*(B-B_mean)).sum().sum()
    corr2 = cov/torch.sqrt(A_var*B_var)
    with open('corr.txt', 'a+') as f:
        f.write(str(corr2))
        f.write('\n')
    return corr2
def calclulate_correlation(input, target):
    """
    calculate the correlation coefficient between input and target.
    The input size must be equal to target(4dim,3dim,2dim are ok) 
    The type of input must be torch.Tensor, channel first.
    """
    # assert isinstance(input, torch.FloatTensor)
    # assert isinstance(target, torch.FloatTensor)
    shape = input.size()
    if len(shape) == 1:
        raise ValueError
    corr = 0
    if len(shape) == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                corr += corr2(input[i,j], target[i,j])
        corr /= (shape[0]*shape[1])
    elif len(shape) == 3:
        for i in range(shape[0]):
            corr += corr2(input[i], target[i])
        corr /= (shape[0])
    else:
        corr = corr2(input, target)
    return corr


if __name__ == '__main__':
    a = torch.tensor([[0.6702,0.3568],[0.6150,0.7474]]).cuda()
    b = torch.tensor([[0.8697,0.2706],[0.6722,0.6825]]).cuda()
    print(calclulate_correlation(a,b))
    


                
