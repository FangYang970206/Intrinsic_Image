import torch
from torch.autograd import Variable



def clc_Loss_albedo(fake,real,device=None):
    lambda_tv = 1e-4

    #prob = (1 - (-3.1416*((fake-real)**2)).exp() )**2

    #loss_data = (prob*( (fake-real)**2 )).mean()
    loss_data = clc_Loss_data(fake,real, device) + lambda_tv*clc_tv_norm(fake)
    #loss_data = prob.mean()
    return loss_data#criterion(fake,real)#clc_Loss_data(fake,real) #+ lambda_tv*clc_tv_norm(fake)


def clc_Loss_shading(fake,real,device=None):

    lambda_tv = 1e-4

    loss_data =  clc_Loss_data(fake,real, device) + lambda_tv*clc_tv_norm(fake)
    return loss_data

def clc_Loss_data(fake,real, device=None):
    # need to optimize !
    weights,neighbors = get_shift_weight(real, device)
    space_weight = [0.0838,0.0838,0.0838,0.0838,0.0113,0.0113,0.0113,0.0113,0.6193]

    tmp_weights = []
    for i in range(len(space_weight)):
        tmp_weights.append(weights[i]*space_weight[i])

    tmp_sum = sum(tmp_weights)
    for i,x in enumerate(tmp_weights):
        tmp_weights[i] = x/tmp_sum

    loss_up     = tmp_weights[0]   * ( (fake-neighbors[0])**2  )
    loss_down   = tmp_weights[1]   * ( (fake-neighbors[1])**2  )
    loss_le     = tmp_weights[2]   * ( (fake-neighbors[2])**2  )
    loss_ri     = tmp_weights[3]   * ( (fake-neighbors[3])**2  )
    loss_ul     = tmp_weights[4]   * ( (fake-neighbors[4])**2  )
    loss_ur     = tmp_weights[5]   * ( (fake-neighbors[5])**2  )
    loss_dl     = tmp_weights[6]   * ( (fake-neighbors[6])**2  )
    loss_dr     = tmp_weights[7]   * ( (fake-neighbors[7])**2  )

    loss_center = tmp_weights[8]* ((fake - real)**2)
    loss = loss_up + loss_down + loss_le + loss_ri + loss_center + loss_ul + loss_ur + loss_dl + loss_dr

    return loss.mean()


def get_shift_index(num,shift_value, device=None):# shift_value = -1 e.g.
    index = [min(max(0,x+shift_value),num-1) for x in range(num)]
    return Variable(torch.LongTensor(index).to(device))

def get_shift_weight(real, device=None):
    sz = real.size()
    ind_up   =  get_shift_index(sz[2], -1, device)
    ind_down =  get_shift_index(sz[2], 1, device)
    ind_le   =  get_shift_index(sz[3], -1, device)
    ind_ri   =  get_shift_index(sz[3], 1, device)

    real_up   = (torch.index_select(real,2,ind_up))
    real_down = (torch.index_select(real,2,ind_down))
    real_le   = (torch.index_select(real,3,ind_le))
    real_ri   = (torch.index_select(real,3,ind_ri))
    real_ul   = (torch.index_select(real_up,3,ind_le))
    real_ur   = (torch.index_select(real_up,3,ind_ri))
    real_dl   = (torch.index_select(real_down,3,ind_le))
    real_dr   = (torch.index_select(real_down,3,ind_ri))

    sigma_color = (((real**2).sum())*real.nelement() - (real.sum())**2).div(real.nelement() ** 2)  # adaptive BF

    gauss_sigma_color = -0.5/(sigma_color*sigma_color)

    gauss_sigma_color = gauss_sigma_color.repeat(real.size())

    w_up   =  (((real - real_up)**2)*(gauss_sigma_color)).exp()
    w_down =  (((real - real_down)**2)*(gauss_sigma_color)).exp()
    w_le   =  (((real - real_le)**2)*(gauss_sigma_color)).exp()
    w_ri   =  (((real - real_ri)**2)*(gauss_sigma_color)).exp()
    w_ul   =  (((real - real_ul)**2)*(gauss_sigma_color)).exp()
    w_ur   =  (((real - real_ur)**2)*(gauss_sigma_color)).exp()
    w_dl   =  (((real - real_dl)**2)*(gauss_sigma_color)).exp()
    w_dr   =  (((real - real_dr)**2)*(gauss_sigma_color)).exp()
    w_center = (((real - real)**2)*(gauss_sigma_color)).exp()



    weights   = [w_up,w_down,w_le,w_ri,w_ul,w_ur,w_dl,w_dr,w_center]
    tmp_sum = sum(weights)


    for i,x in enumerate(weights):
        weights[i] = x/tmp_sum

    neighbors = [real_up,real_down,real_le,real_ri,real_ul,real_ur,real_dl,real_dr]

    return weights,neighbors



def clc_tv_norm(input):
    return torch.mean(torch.abs(input[:,:,:,:-1]-input[:,:,:,1:])) + torch.mean(torch.abs(input[:,:,:-1,:]-input[:,:,1:,:]))
