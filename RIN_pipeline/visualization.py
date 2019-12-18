import os, math, torch, torch.nn as nn, torchvision.utils, numpy as np, scipy.misc, pdb
from torch.autograd import Variable
from tqdm import tqdm
import pipeline
import sys
from .eval_meterics import calc_siError, clc_pad


def visualize_reflection(model, loader, device, save_path):
        model.eval()
        
        inputs = []
        predictions = []
        targets = []
        criterion = nn.MSELoss(size_average=True).to(device)
        val_loss = 0
        for ind, tensors in enumerate(loader):

            inp = [ t.float().to(device) for t in tensors]
            lab_inp, lab_refl_targ = inp
            refl_pred = model.forward(lab_inp)

            val_loss += criterion(refl_pred, lab_refl_targ).item()
            # predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
            inputs.extend([img.squeeze() for img in lab_inp.data.split(1)])
            predictions.extend([img.squeeze() for img in refl_pred.data.split(1)])
            targets.extend([img.squeeze() for img in lab_refl_targ.data.split(1)])

        val_loss = val_loss/float(ind+1)
        # print len(inputs), len(predictions), len(targets)
        images = [[inputs[i], predictions[i], targets[i]] for i in range(len(inputs))]
        images = [img for sublist in images for img in sublist]
        
        grid = torchvision.utils.make_grid(images, nrow=3, padding=0).cpu().numpy().transpose(1,2,0)
        grid = np.clip(grid, 0, 1)
        scipy.misc.imsave(save_path, grid)
        
        return val_loss

def visualize_shader(model, loader, device, save_path):
        model.eval()
        
        inputs = []
        predictions = []
        targets = []
        criterion = nn.MSELoss(size_average=True).to(device)
        val_loss = 0
        for ind, tensors in enumerate(loader):

            inp = [t.float().to(device) for t in tensors]
            lab_inp, lab_sha_targ = inp
            sha_pred = model.forward(lab_inp)

            val_loss += criterion(sha_pred, lab_sha_targ).item()
            # predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
            inputs.extend([img.squeeze() for img in lab_inp.data.split(1)])
            predictions.extend([img.repeat(1,3,1,1).squeeze() for img in sha_pred.data.split(1)])
            targets.extend([img.repeat(1,3,1,1).squeeze() for img in lab_sha_targ.data.split(1)])

        val_loss = val_loss/float(ind+1)
        # print len(inputs), len(predictions), len(targets)
        images = [[inputs[i], predictions[i], targets[i]] for i in range(len(inputs))]
        images = [img for sublist in images for img in sublist]
        
        grid = torchvision.utils.make_grid(images, nrow=3, padding=0).cpu().numpy().transpose(1,2,0)
        grid = np.clip(grid, 0, 1)
        scipy.misc.imsave(save_path, grid)
        
        return val_loss

def visualize_composer(model, loader, device, save_path):
        model.eval()
        
        inputs = []
        refl_target = []
        sha_target = []
        refl_predictions = []
        sha_predictions = []
        reconstructions = []
        shape_predictions = []
        
        criterion = nn.MSELoss(size_average=True).to(device)
        loss = []
        recon_loss, refl_loss, sha_loss = 0, 0, 0
        with torch.no_grad():
            for ind, tensors in enumerate(loader):
            
                inp = [t.float().to(device) for t in tensors]
                lab_inp, lab_refl_targ, lab_sha_targ, mask = inp
                recon_pred, refl_pred,  sha_pred, shape_pred = model.forward(lab_inp)
                # recon_pred, refl_pred,  sha_pred = model.forward(lab_inp)

                recon_pred, refl_pred, sha_pred, shape_pred = recon_pred*mask, refl_pred*mask, sha_pred*mask, shape_pred*mask
                lab_inp, lab_refl_targ, lab_sha_targ = lab_inp*mask, lab_refl_targ*mask, lab_sha_targ*mask
                recon_loss += criterion(recon_pred, lab_inp).item()
                refl_loss += criterion(refl_pred, lab_refl_targ).item()
                sha_loss += criterion(sha_pred, lab_sha_targ).item()
    
                # predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
                inputs.extend([img.squeeze() for img in lab_inp.data.split(1)])
                refl_target.extend([img.squeeze() for img in lab_refl_targ.data.split(1)])
                # sha_target.extend([img.repeat(1,3,1,1).squeeze() for img in lab_sha_targ.data.split(1)])
                sha_target.extend([img.squeeze() for img in lab_sha_targ.data.split(1)])
                reconstructions.extend([img.squeeze() for img in recon_pred.data.split(1)])
                refl_predictions.extend([img.squeeze() for img in refl_pred.data.split(1)])
                sha_predictions.extend([img.squeeze() for img in sha_pred.data.split(1)])
                # sha_predictions.extend([img.repeat(1,3,1,1).squeeze() for img in sha_pred.data.split(1)])
                # shape_predictions.extend([img.squeeze() for img in shape_pred.data.split(1)])
    
            recon_loss = recon_loss/float(ind+1)
            refl_loss = refl_loss/float(ind+1)
            sha_loss = sha_loss/float(ind+1)
            loss.append(recon_loss)
            loss.append(refl_loss)
            loss.append(sha_loss)
            # print len(inputs), len(predictions), len(targets)
            images = [[inputs[i], refl_target[i], sha_target[i], reconstructions[i], refl_predictions[i], sha_predictions[i]] for i in range(len(inputs))]
            images = [img for sublist in images for img in sublist]
            
            grid = torchvision.utils.make_grid(images, nrow=6, padding=0).cpu().numpy().transpose(1,2,0)
            grid = np.clip(grid, 0, 1)
            scipy.misc.imsave(save_path, grid)
        
        return loss

def visualize_shapenet(model, loader, device, save_path):
        model.eval()
        
        inputs = []
        refl_target = []
        sha_target = []
        shape_target = []
        refl_predictions = []
        sha_predictions = []
        reconstructions = []
        shape_predictions = []
        
        # criterion = nn.MSELoss(size_average=True).to(device)
        loss = []
        # recon_loss, refl_loss, sha_loss = 0, 0, 0
        with torch.no_grad():
            for ind, tensors in enumerate(loader):
            
                inp = [t.float().to(device) for t in tensors]
                lab_inp, lab_refl_targ, lab_sha_targ, lab_shape_targ = inp
                recon_pred, refl_pred,  sha_pred, shape_pred = model.forward(lab_inp)
    
                # recon_loss += criterion(recon_pred, lab_inp).item()
                # refl_loss += criterion(refl_pred, lab_refl_targ).item()
                # sha_loss += criterion(sha_pred, lab_sha_targ).item()
    
                # predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
                inputs.extend([img.squeeze() for img in lab_inp.data.split(1)])
                refl_target.extend([img.squeeze() for img in lab_refl_targ.data.split(1)])
                sha_target.extend([img.repeat(1,3,1,1).squeeze() for img in lab_sha_targ.data.split(1)])
                shape_target.extend([img.squeeze() for img in lab_shape_targ.data.split(1)])
                reconstructions.extend([img.squeeze() for img in recon_pred.data.split(1)])
                refl_predictions.extend([img.squeeze() for img in refl_pred.data.split(1)])
                sha_predictions.extend([img.repeat(1,3,1,1).squeeze() for img in sha_pred.data.split(1)])
                shape_predictions.extend([img.squeeze() for img in shape_pred.data.split(1)])
    
            # recon_loss = recon_loss/float(ind+1)
            # refl_loss = refl_loss/float(ind+1)
            # sha_loss = sha_loss/float(ind+1)
            # loss.append(recon_loss)
            # loss.append(refl_loss)
            # loss.append(sha_loss)
            # print len(inputs), len(predictions), len(targets)
            images = [[inputs[i], refl_target[i], sha_target[i], shape_target[i], reconstructions[i], refl_predictions[i], sha_predictions[i], shape_predictions[i]] for i in range(len(inputs))]
            images = [img for sublist in images for img in sublist]
            
            grid = torchvision.utils.make_grid(images, nrow=8, padding=0).cpu().numpy().transpose(1,2,0)
            grid = np.clip(grid, 0, 1)
            scipy.misc.imsave(save_path, grid)
        
        return loss

def visualize_MIT(model, loader, device, save_path):
        model.eval()
        criterion = nn.MSELoss(size_average=True).to(device)
        inputs = []
        refl_target = []
        sha_target = []
        # shape_target = []
        refl_predictions = []
        sha_predictions = []
        reconstructions = []
        shape_predictions = []
        recon_loss = refl_loss = sha_loss =0
        loss = []
        with torch.no_grad():
            for ind, tensors in enumerate(loader):
            
                inp = [t.float().to(device) for t in tensors]
                lab_inp, lab_refl_targ, lab_sha_targ, lab_mask_targ = inp
                recon_pred, refl_pred,  sha_pred, shape_pred = model.forward(lab_inp)

                recon_pred = recon_pred * lab_mask_targ
                refl_pred = refl_pred * lab_mask_targ
                sha_pred = sha_pred * lab_mask_targ[:, 1, :, :].unsqueeze(1)

                recon_loss += criterion(recon_pred, lab_inp).item()
                refl_loss += criterion(refl_pred, lab_refl_targ).item()
                sha_loss += criterion(sha_pred, lab_sha_targ).item()
    
                # predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
                inputs.extend([img.squeeze() for img in lab_inp.data.split(1)])
                refl_target.extend([img.squeeze() for img in lab_refl_targ.data.split(1)])
                sha_target.extend([img.repeat(1,3,1,1).squeeze() for img in lab_sha_targ.data.split(1)])
                # shape_target.extend([img.squeeze() for img in lab_shape_targ.data.split(1)])
                reconstructions.extend([img.squeeze() for img in recon_pred.data.split(1)])
                refl_predictions.extend([img.squeeze() for img in refl_pred.data.split(1)])
                sha_predictions.extend([img.repeat(1,3,1,1).squeeze() for img in sha_pred.data.split(1)])
                shape_predictions.extend([img.squeeze() for img in shape_pred.data.split(1)])
    
            recon_loss = recon_loss/float(ind+1)
            refl_loss = refl_loss/float(ind+1)
            sha_loss = sha_loss/float(ind+1)
            loss.append(recon_loss)
            loss.append(refl_loss)
            loss.append(sha_loss)
            
            images = [[inputs[i], refl_target[i], sha_target[i], reconstructions[i], refl_predictions[i], sha_predictions[i], shape_predictions[i]] for i in range(len(inputs))]
            images = [img for sublist in images for img in sublist]
            
            grid = torchvision.utils.make_grid(images, nrow=7, padding=0).cpu().numpy().transpose(1,2,0)
            grid = np.clip(grid, 0, 1)
            scipy.misc.imsave(save_path, grid)
        
        return loss

def visualize_MPI(model, loader, device, save_path):
        model.eval()
        # h, w = 436,1024
        # pad_h,pad_w = clc_pad(h,w,32)
        # tmp_pad = nn.ReflectionPad2d((0,pad_w,0,pad_h))
        # tmp_inversepad = nn.ReflectionPad2d((0,-pad_w,0,-pad_h))
        inputs = []
        refl_target = []
        shad_target = []
        refl_predictions = []
        shad_predictions = []
        reconstructions = []
        shape_predictions = []
        with torch.no_grad():
            for ind, tensors in enumerate(loader):
            
                inp = [t.to(device) for t in tensors]
                input_g, albedo_g, shading_g, mask_g = inp

                # input_g, albedo_g, shading_g, mask_g = tmp_pad(input_g),tmp_pad(albedo_g),tmp_pad(shading_g),tmp_pad(mask_g)
                _, albedo_fake, shading_fake, shape_fake = model.forward(input_g)
                # input_g,albedo_g,shading_g,mask_g = tmp_inversepad(input_g),tmp_inversepad(albedo_g),tmp_inversepad(shading_g),tmp_inversepad(mask_g)
                # input_fake, albedo_fake, shading_fake  = tmp_inversepad(input_fake.clamp(0,1)), tmp_inversepad(albedo_fake.clamp(0,1)),tmp_inversepad(shading_fake.clamp(0,1))
                
                albedo_fake  = albedo_fake*mask_g
                # input_fake = albedo_fake * shading_fake.repeat(1,3,1,1)
                input_fake = albedo_fake * shading_fake
                if ind < 10:
                    inputs.extend([img.squeeze() for img in input_g.data.split(1)])
                    refl_target.extend([img.squeeze() for img in albedo_g.data.split(1)])
                    # shad_target.extend([img.repeat(1,3,1,1).squeeze() for img in shading_g.data.split(1)])
                    shad_target.extend([img.squeeze() for img in shading_g.data.split(1)])
                    reconstructions.extend([img.squeeze() for img in input_fake.data.split(1)])
                    refl_predictions.extend([img.squeeze() for img in albedo_fake.data.split(1)])
                    # shad_predictions.extend([img.repeat(1,3,1,1).squeeze() for img in shading_fake.data.split(1)])
                    shad_predictions.extend([img.squeeze() for img in shading_fake.data.split(1)])
                    shape_predictions.extend([img.squeeze() for img in shape_fake.data.split(1)])
                else:
                    break
            images = [[inputs[i], refl_target[i], shad_target[i], reconstructions[i], refl_predictions[i], shad_predictions[i], shape_predictions[i]] for i in range(len(inputs))]
            images = [img for sublist in images for img in sublist]
            
            grid = torchvision.utils.make_grid(images, nrow=7, padding=0).cpu().numpy().transpose(1,2,0)
            grid = np.clip(grid, 0, 1)
            scipy.misc.imsave(save_path, grid)




