import os, math, torch, torch.nn as nn, torchvision.utils, numpy as np, scipy.misc
from torch.autograd import Variable
from tqdm import tqdm
from .eval_meterics import calc_siError, clc_pad
from .whdr import compute_whdr
import json


def MPI_test(model, loader, device):
        model.eval()
        # h, w = 436,1024
        # pad_h,pad_w = clc_pad(h,w,32)
        # tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        # tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
        # print('eval')
        criterion = nn.MSELoss(size_average=True).to(device)
        count = 0
        A_mse, S_mse = 0, 0
        with torch.no_grad():
            for _, tensors in enumerate(loader):
            
                inp = [t.to(device) for t in tensors]
                input_g, albedo_g, shading_g, mask_g = inp
                # print('test here')
                # input_g, albedo_g, shading_g, mask_g = tmp_pad(input_g),tmp_pad(albedo_g),tmp_pad(shading_g),tmp_pad(mask_g)
                _, albedo_fake, shading_fake, _ = model.forward(input_g)
                # print('forward success')
                # input_g,albedo_g,shading_g,mask_g = tmp_inversepad(input_g),tmp_inversepad(albedo_g),tmp_inversepad(shading_g),tmp_inversepad(mask_g)
                # input_fake, albedo_fake, shading_fake  = tmp_inversepad(input_fake.clamp(0,1)), tmp_inversepad(albedo_fake.clamp(0,1)),tmp_inversepad(shading_fake.clamp(0,1))
                
                albedo_fake  = albedo_fake*mask_g

                A_mse += criterion(albedo_fake, albedo_g).item()
                S_mse += criterion(shading_fake, shading_g).item()

                # A_siMSE,A_siLMSE,A_DSSIM, batch_channel1 = calc_siError(albedo_fake,albedo_g,mask_g)
                # S_siMSE,S_siLMSE,S_DSSIM, batch_channel2 = calc_siError(shading_fake,shading_g,None)
                
                # A_simse += A_siMSE/batch_channel1
                # A_silmse += A_siLMSE/batch_channel1
                # A_dssim += A_DSSIM/batch_channel1
                # S_simse += S_siMSE/batch_channel2
                # S_silmse += S_siLMSE/batch_channel2
                # S_dssim += S_DSSIM/batch_channel2
                count += 1
        return [A_mse/count, S_mse/count]

def MPI_test_remove_shape(model, loader, device):
        model.eval()
        # h, w = 436,1024
        # pad_h,pad_w = clc_pad(h,w,32)
        # tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        # tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
        # print('eval')
        criterion = nn.MSELoss(size_average=True).to(device)
        count = 0
        A_mse, S_mse = 0, 0
        with torch.no_grad():
            for _, tensors in enumerate(loader):
            
                inp = [t.to(device) for t in tensors]
                input_g, albedo_g, shading_g, mask_g = inp
                # print('test here')
                # input_g, albedo_g, shading_g, mask_g = tmp_pad(input_g),tmp_pad(albedo_g),tmp_pad(shading_g),tmp_pad(mask_g)
                _, albedo_fake, shading_fake = model.forward(input_g)
                # print('forward success')
                # input_g,albedo_g,shading_g,mask_g = tmp_inversepad(input_g),tmp_inversepad(albedo_g),tmp_inversepad(shading_g),tmp_inversepad(mask_g)
                # input_fake, albedo_fake, shading_fake  = tmp_inversepad(input_fake.clamp(0,1)), tmp_inversepad(albedo_fake.clamp(0,1)),tmp_inversepad(shading_fake.clamp(0,1))
                
                albedo_fake  = albedo_fake*mask_g

                A_mse += criterion(albedo_fake, albedo_g).item()
                S_mse += criterion(shading_fake, shading_g).item()

                # A_siMSE,A_siLMSE,A_DSSIM, batch_channel1 = calc_siError(albedo_fake,albedo_g,mask_g)
                # S_siMSE,S_siLMSE,S_DSSIM, batch_channel2 = calc_siError(shading_fake,shading_g,None)
                
                # A_simse += A_siMSE/batch_channel1
                # A_silmse += A_siLMSE/batch_channel1
                # A_dssim += A_DSSIM/batch_channel1
                # S_simse += S_siMSE/batch_channel2
                # S_silmse += S_siLMSE/batch_channel2
                # S_dssim += S_DSSIM/batch_channel2
                count += 1
        return [A_mse/count, S_mse/count]

def MPI_test_unet(model, loader, device, refl_multi_size=None, shad_multi_size=None):
        model.eval()
        # h, w = 436,1024
        # pad_h,pad_w = clc_pad(h,w,32)
        # tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        # tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
        # print('eval')
        criterion = nn.MSELoss(size_average=True).to(device)
        count = 0
        A_mse, S_mse = 0, 0
        with torch.no_grad():
            for _, tensors in enumerate(loader):
            
                inp = [t.to(device) for t in tensors]
                input_g, albedo_g, shading_g, mask_g = inp
                # print('test here')
                # input_g, albedo_g, shading_g, mask_g = tmp_pad(input_g),tmp_pad(albedo_g),tmp_pad(shading_g),tmp_pad(mask_g)
                if refl_multi_size and shad_multi_size:
                    albedo_fake, shading_fake, _, _ = model.forward(input_g)
                elif refl_multi_size or shad_multi_size:
                    albedo_fake, shading_fake, _ = model.forward(input_g)
                else:
                    albedo_fake, shading_fake = model.forward(input_g)
                # print('forward success')
                # input_g,albedo_g,shading_g,mask_g = tmp_inversepad(input_g),tmp_inversepad(albedo_g),tmp_inversepad(shading_g),tmp_inversepad(mask_g)
                # input_fake, albedo_fake, shading_fake  = tmp_inversepad(input_fake.clamp(0,1)), tmp_inversepad(albedo_fake.clamp(0,1)),tmp_inversepad(shading_fake.clamp(0,1))
                
                albedo_fake  = albedo_fake*mask_g

                A_mse += criterion(albedo_fake, albedo_g).item()
                S_mse += criterion(shading_fake, shading_g).item()

                # A_siMSE,A_siLMSE,A_DSSIM, batch_channel1 = calc_siError(albedo_fake,albedo_g,mask_g)
                # S_siMSE,S_siLMSE,S_DSSIM, batch_channel2 = calc_siError(shading_fake,shading_g,None)
                
                # A_simse += A_siMSE/batch_channel1
                # A_silmse += A_siLMSE/batch_channel1
                # A_dssim += A_DSSIM/batch_channel1
                # S_simse += S_siMSE/batch_channel2
                # S_silmse += S_siLMSE/batch_channel2
                # S_dssim += S_DSSIM/batch_channel2
                count += 1
        return [A_mse/count, S_mse/count]

def IIW_test_unet(model, loader, device):
        model.eval()
        count = 0
        score = 0
        with torch.no_grad():
            for _, tensors in enumerate(loader):

                input_g, label_txt = tensors
                label_txt = label_txt[0]
                _, _, h, w = input_g.size()
                pad_h,pad_w = clc_pad(h,w,16)
                input_g = input_g.to(device)
                tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
                tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
                input_g = tmp_pad(input_g)
                albedo_fake = model.forward(input_g)
                albedo_fake = tmp_inversepad(albedo_fake)
                albedo_pred = albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
                label_json = label_txt.replace('txt', 'json')
                score += compute_whdr(albedo_pred, json.load(open(label_json)), 0.1)
                count += 1
        return score/count

def MPI_test_unet_one(model, loader, device, choose='refl'):
        model.eval()
        # h, w = 436,1024
        # pad_h,pad_w = clc_pad(h,w,32)
        # tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        # tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
        # print('eval')
        criterion = nn.MSELoss(size_average=True).to(device)
        count = 0
        A_mse, S_mse = 0, 0
        with torch.no_grad():
            for _, tensors in enumerate(loader):
            
                inp = [t.to(device) for t in tensors]
                input_g, albedo_g, shading_g, mask_g = inp
                # print('test here')
                # input_g, albedo_g, shading_g, mask_g = tmp_pad(input_g),tmp_pad(albedo_g),tmp_pad(shading_g),tmp_pad(mask_g)
                if choose == 'refl':
                    albedo_fake = model.forward(input_g)
                    albedo_fake  = albedo_fake*mask_g
                    A_mse += criterion(albedo_fake, albedo_g).item()
                else:
                    shading_fake = model.forward(input_g)
                    S_mse += criterion(shading_fake, shading_g).item()
                # print('forward success')
                # input_g,albedo_g,shading_g,mask_g = tmp_inversepad(input_g),tmp_inversepad(albedo_g),tmp_inversepad(shading_g),tmp_inversepad(mask_g)
                # input_fake, albedo_fake, shading_fake  = tmp_inversepad(input_fake.clamp(0,1)), tmp_inversepad(albedo_fake.clamp(0,1)),tmp_inversepad(shading_fake.clamp(0,1))
                # A_siMSE,A_siLMSE,A_DSSIM, batch_channel1 = calc_siError(albedo_fake,albedo_g,mask_g)
                # S_siMSE,S_siLMSE,S_DSSIM, batch_channel2 = calc_siError(shading_fake,shading_g,None)
                
                # A_simse += A_siMSE/batch_channel1
                # A_silmse += A_siLMSE/batch_channel1
                # A_dssim += A_DSSIM/batch_channel1
                # S_simse += S_siMSE/batch_channel2
                # S_silmse += S_siLMSE/batch_channel2
                # S_dssim += S_DSSIM/batch_channel2
                count += 1
        return A_mse/count if A_mse != 0 else S_mse/count