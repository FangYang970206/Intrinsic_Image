import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim


def calc_siError(fake_in,real_in,mask_in):

    if mask_in is None:
        mask_in = (Variable(torch.ones(fake_in.size()))).cuda()
    tmp_mse,batch_ch  = clc_efficient_siMSE(fake_in,real_in,mask_in)
    tmp_lmse,batch_ch  = clc_efficient_siLMSE(fake_in,real_in,mask_in)
    tmp_dssim,batch_ch = clc_efficient_DSSIM(fake_in,real_in,mask_in)
    return tmp_mse,tmp_lmse,tmp_dssim,batch_ch

def clc_efficient_siMSE(fake,real,mask):
    B,C,H,W = fake.size()
    bn_ch = B*C ; hw = H*W
    if isinstance(fake, Variable):
        X = (mask*fake).view(bn_ch,H,W).data
        Y = (mask*real).view(bn_ch,H,W).data
        M =  mask.view(bn_ch,H,W).data
    else:
        X = (mask*fake).view(bn_ch,H,W)
        Y = (mask*real).view(bn_ch,H,W)
        M =  mask.view(bn_ch,H,W)
    mse_error = 0.0
    for bc in range(bn_ch):
        if torch.sum(M[bc,:,:])==0:
            continue
        deno = torch.sum(X[bc,:,:]**2)
        nume = torch.sum(Y[bc,:,:]*X[bc,:,:])
        if deno>1e-5:
            alpha = nume/deno
        else:
            alpha = 0
        mse_error += torch.mean((((X[bc,:,:]*alpha) - Y[bc,:,:])**2))
    return mse_error,bn_ch

def clc_efficient_siLMSE(fake,real,mask):

    B,C,H,W = fake.size()
    st = int(W//10)
    half_st = int(st // 2)

    pad_h,pad_w = clc_pad(H,W,half_st)

    tmp_pad     = nn.ZeroPad2d((0,pad_w,0,pad_h))
    tmp_unpad   = nn.ZeroPad2d((0,-pad_w,0,-pad_h))

    X = tmp_pad(fake*mask)
    Y = tmp_pad(mask*real)
    M = tmp_pad(mask)

    idx_jn = (H+pad_h)//half_st
    idx_in = (W+pad_w)//half_st

    LMSE_error = 0
    count = 0

    X_ij = torch.zeros(B,C,half_st*2,half_st*2).cuda()
    Y_ij = torch.zeros(B,C,half_st*2,half_st*2).cuda()
    M_ij = torch.zeros(B,C,half_st*2,half_st*2).cuda()

    for j in range(idx_jn-2):
        for i in range(idx_in-2):
            X_tmp = X[:,:,j*half_st:(j+2)*half_st,i*half_st:(i+2)*half_st]
            Y_tmp = Y[:,:,j*half_st:(j+2)*half_st,i*half_st:(i+2)*half_st]
            M_tmp = M[:,:,j*half_st:(j+2)*half_st,i*half_st:(i+2)*half_st]

            X_ij.copy_(X_tmp.data)

            Y_ij.copy_(Y_tmp.data)
            M_ij.copy_(M_tmp.data)
            batch_error,_ = clc_efficient_siMSE(X_ij,Y_ij,M_ij)
            count += 1
            LMSE_error += batch_error

    return LMSE_error/(count),B*C


def clc_efficient_DSSIM(fake,real,mask):

    B,C,H,W = fake.size()

    bn_ch = B*C

    X = (mask*fake).view(bn_ch,H,W)
    Y = (mask*real).view(bn_ch,H,W)
    M =  mask.view(bn_ch,H,W)

    X = np.transpose(X.data.cpu().numpy(),(1,2,0))
    Y = np.transpose(Y.data.cpu().numpy(),(1,2,0))
    M = np.transpose(M.data.cpu().numpy(),(1,2,0))

    s = 0.0

    for i in range(bn_ch):
        s += (1-ssim(Y[:,:,i],X[:,:,i]))/2

    return s,bn_ch

def clc_pad(h,w,st=32):## default st--> 32
    def _f(s):
        n = s//st
        r = s %st
        if r == 0:
            return 0
        else:
            return st-r
    return _f(h),_f(w)

def clc_unpad(X,pad_h,pad_w):
    h = X.size()[2] - pad_h
    w = X.size()[3] - pad_w
    return X[:,:,:h,:w]