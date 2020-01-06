import os, math, torch, torch.nn as nn, torchvision.utils, numpy as np, scipy.misc
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm
from .eval_meterics import calc_siError, clc_pad
from .whdr import compute_whdr
import json
from utils import *


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

def evaluate_one_k(output, label):
    b, c, h, w = output.size()
    mask = ~torch.isnan(output)
    output = output * mask.float()
    label = label * mask.float()
    v_output = output.view(1, -1)
    v_label = label.view(1, -1)
    k = (v_output.mm(v_label.transpose(0, 1))) / (v_output.mm(v_output.transpose(0, 1)))
    tmp = torch.min(k * v_output, torch.ones_like(v_output)) - v_label
    error = tmp.mm(tmp.transpose(0, 1)) / (b * c * h * w)
    return error

def MIT_error(output, label, mask):
    output, label = output * mask, label * mask
    alpha = torch.sum(output * label) / torch.max(torch.tensor([1e-8]), torch.sum(output * output))
    output = alpha * output
    error = torch.mean((torch.masked_select(output, mask.ge(0.5)) - torch.masked_select(label, mask.ge(0.5))) ** 2)
    return error

def MIT_test_unet(model, loader, device, args):
    model.eval()
    refl_multi_size = args.refl_multi_size
    shad_multi_size = args.shad_multi_size
    count = 0
    A_mse, S_mse = 0, 0
    with torch.no_grad():
        for ind, tensors in enumerate(loader):
            print(ind)
            inp = [t.to(device) for t in tensors]
            input_g, albedo_g, shading_g, mask_g = inp
            
            if refl_multi_size and shad_multi_size:
                albedo_fake, shading_fake, _, _ = model.forward(input_g)
            elif refl_multi_size or shad_multi_size:
                albedo_fake, shading_fake, _ = model.forward(input_g)
            else:
                albedo_fake, shading_fake = model.forward(input_g)

            albedo_fake  = albedo_fake*mask_g
            shading_fake = shading_fake*mask_g

            albedo_fake = albedo_fake.cpu().clamp(0, 1).squeeze()
            shading_fake = shading_fake.cpu().clamp(0, 1).squeeze()
            albedo_g = albedo_g.cpu().clamp(0, 1).squeeze()
            shading_g = shading_g.cpu().clamp(0, 1).squeeze()
            mask_g = mask_g.cpu().clamp(0, 1).squeeze()

            A_mse += MIT_error(albedo_fake, albedo_g, mask_g).item()
            S_mse += MIT_error(shading_fake, shading_g, mask_g).item()
            count += 1
    return [A_mse/count, S_mse/count]

def MPI_test_unet(model, loader, device, args):
    model.eval()
    refl_multi_size = args.refl_multi_size
    shad_multi_size = args.shad_multi_size
    fullsize = args.fullsize_test
    # ToPIL = transforms.ToPILImage()
    # h, w = 436,1024
    # pad_h,pad_w = clc_pad(h,w,32)
    # tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
    # tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
    # print('eval')
    # criterion = nn.MSELoss(size_average=True).to(device)
    if fullsize:
        h, w = 436,1024
        pad_h,pad_w = clc_pad(h,w,16)
        print(pad_h, pad_w)
        tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
    count = 0
    A_mse, S_mse = 0, 0
    with torch.no_grad():
        for ind, tensors in enumerate(loader):
            print(ind)
            inp = [t.to(device) for t in tensors]
            input_g, albedo_g, shading_g, mask_g = inp
            
            if fullsize:
                input_g = tmp_pad(input_g)
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
            if fullsize:
                albedo_fake, shading_fake = tmp_inversepad(albedo_fake), tmp_inversepad(shading_fake)

            albedo_fake  = albedo_fake*mask_g

            albedo_fake = albedo_fake.cpu().clamp(0, 1)
            shading_fake = shading_fake.cpu().clamp(0, 1)
            albedo_g = albedo_g.cpu().clamp(0, 1)
            shading_g = shading_g.cpu().clamp(0, 1)

            A_mse += evaluate_one_k(albedo_fake, albedo_g).item()
            S_mse += evaluate_one_k(shading_fake, shading_g).item()

            # lab_refl_targ = ToPIL(albedo_g.squeeze())
            # lab_sha_targ = ToPIL(shading_g.squeeze())
            # refl_pred = ToPIL(albedo_fake.squeeze())
            # sha_pred = ToPIL(shading_fake.squeeze())

            # check_folder(os.path.join(args.save_path, "refl_target_test3"))
            # check_folder(os.path.join(args.save_path, "refl_output_test3"))
            # check_folder(os.path.join(args.save_path, "shad_target_test3"))
            # check_folder(os.path.join(args.save_path, "shad_output_test3"))

            # lab_refl_targ.save(os.path.join(args.save_path, "refl_target_test3", "{}.png".format(ind)))
            # lab_sha_targ.save(os.path.join(args.save_path, "shad_target_test3", "{}.png".format(ind)))
            # refl_pred.save(os.path.join(args.save_path, "refl_output_test3", "{}.png".format(ind)))
            # sha_pred.save(os.path.join(args.save_path, "shad_output_test3", "{}.png".format(ind)))
            
            # lab_refl_targ.save(os.path.join(args.save_path, "refl_target_test2", "{}.png".format(ind)), quality=95)
            # lab_sha_targ.save(os.path.join(args.save_path, "shad_target_test2", "{}.png".format(ind)), quality=95)
            # refl_pred.save(os.path.join(args.save_path, "refl_output_test2", "{}.png".format(ind)), quality=95)
            # sha_pred.save(os.path.join(args.save_path, "shad_output_test2", "{}.png".format(ind)),  quality=95)

            # lab_refl_targ = albedo_g.squeeze().cpu().numpy().transpose(1,2,0)
            # lab_sha_targ = shading_g.squeeze().cpu().numpy().transpose(1,2,0)
            # refl_pred = albedo_fake.squeeze().cpu().numpy().transpose(1,2,0)
            # sha_pred = shading_fake.squeeze().cpu().numpy().transpose(1,2,0)

            # lab_refl_targ = np.clip(lab_refl_targ, 0, 1)
            # lab_sha_targ = np.clip(lab_sha_targ, 0, 1)
            # refl_pred = np.clip(refl_pred, 0, 1)
            # sha_pred = np.clip(sha_pred, 0, 1)

            # check_folder(os.path.join(args.save_path, "refl_target_test"))
            # check_folder(os.path.join(args.save_path, "refl_output_test"))
            # check_folder(os.path.join(args.save_path, "shad_target_test"))
            # check_folder(os.path.join(args.save_path, "shad_output_test"))

            # scipy.misc.imsave(os.path.join(args.save_path, "refl_target_test", "{}.png".format(ind)), lab_refl_targ)
            # scipy.misc.imsave(os.path.join(args.save_path, "refl_output_test", "{}.png".format(ind)), refl_pred)
            # scipy.misc.imsave(os.path.join(args.save_path, "shad_target_test", "{}.png".format(ind)), lab_sha_targ)
            # scipy.misc.imsave(os.path.join(args.save_path, "shad_output_test", "{}.png".format(ind)), sha_pred)
            # A_mse += criterion(albedo_fake, albedo_g).item()
            # S_mse += criterion(shading_fake, shading_g).item()
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