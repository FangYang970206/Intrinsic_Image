import sys, math, numpy as np, pdb
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pytorch_ssim
import pipeline


class UnsupervisionTrainer:
    def __init__(self, model, loader, lr, device, writer, step):
        self.model = model
        self.loader = loader
        self.criterion1 = nn.MSELoss(size_average=True).to(device)
        # self.criterion2 = pytorch_ssim.SSIM()
        self.device = device
        self.writer = writer
        self.step = step
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def __epoch(self):
        self.model.train()

        progress = tqdm( total=len(self.loader.dataset))

        for ind, unlabeled in enumerate(self.loader):
            unlabeled = [t.float().to(self.device) for t in unlabeled]
            
            un_inp, lab_refl_targ, lab_shad_targ = unlabeled
            un_recon_pred, lab_refl_pred, lab_shad_pred, _ = self.model.forward(un_inp)
            

            # self.optimizer2.zero_grad()   
            # # refl_loss_SSIM = -self.criterion2(lab_refl_targ, lab_refl_pred)
            # refl_loss = self.criterion1(lab_refl_pred, lab_refl_targ)
            # refl_loss.backward()
            # self.optimizer2.step()

            # self.optimizer1.zero_grad()
            # shad_loss = self.criterion1(lab_shad_pred, lab_shad_targ)
            # shad_loss.backward()
            # self.optimizer1.step()

            # self.writer.add_scalar('supervision_refl_loss', refl_loss.item(), self.step)
            # self.writer.add_scalar('supervision_shad_loss', shad_loss.item(), self.step)
            # self.step += 1

            self.optimizer.zero_grad()
            refl_loss = self.criterion1(lab_refl_pred, lab_refl_targ)
            shad_loss = self.criterion1(lab_shad_pred, lab_shad_targ)
            un_loss = self.criterion1(un_recon_pred, un_inp)
            un_loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('unsupervision_refl_loss', refl_loss.item(), self.step)
            self.writer.add_scalar('unsupervision_shad_loss', shad_loss.item(), self.step)
            self.writer.add_scalar('unsupervision_un_loss', un_loss.item(), self.step)
            self.step += 1

            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f| %.5f | %.3f' % (un_loss.item(), refl_loss.item(), shad_loss.item()) )
 
        return self.step

    def train(self):
        errors = self.__epoch()
        return errors

if __name__ == '__main__':
    import sys
    sys.path.append('../')




