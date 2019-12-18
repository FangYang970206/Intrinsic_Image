from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


class ComposerTrainer:
    def __init__(self, model, loader, device, optimizer, writer, step):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).to(device)
        self.device = device
        self.optimizer = optimizer
        self.writer = writer
        self.step = step

    def __epoch(self):
        self.model.train()

        progress = tqdm(total=len(self.loader.dataset))

        for _, batch in enumerate(self.loader):
            batch_data = [t.float().to(self.device) for t in batch]

            lab_inp, lab_refl_targ, lab_sha_targ = batch_data
            
            recon_pred, refl_pred,  sha_pred= self.model.forward(lab_inp)
            
            refl_loss = self.criterion(refl_pred, lab_refl_targ).item()
            sha_loss = self.criterion(sha_pred, lab_sha_targ).item()

            self.optimizer.zero_grad()
            recon_loss = self.criterion(recon_pred, lab_inp)
            recon_loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('train_refl_loss', refl_loss, self.step)
            self.writer.add_scalar('train_sha_loss', sha_loss, self.step)
            self.writer.add_scalar('train_recon_loss', recon_loss.item(), self.step)
            self.step += 1
            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f' % (recon_loss.item()))
            
        return self.step

    def train(self):
        return self.__epoch()
        




