import torch
import torch.nn as nn


class GradientLoss(nn.Module):
    def __init__(self, x_flag=True, y_flag=True, device=None):
        super(GradientLoss, self).__init__()
        self.x_flag = x_flag
        self.y_flag = y_flag
        self.l1_loss = nn.L1Loss().to(device)

    def forward(self, pred, targ):
        if pred.size() != targ.size() and pred.dim() == 4:
            raise ValueError('pred and targ should be the same size, and its dim should be 4')
        if self.x_flag:
            pred_grad_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            targ_grad_x = targ[:, :, 1:, :] - targ[:, :, :-1, :]
        if self.y_flag:
            pred_grad_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            targ_grad_y = targ[:, :, :, 1:] - targ[:, :, :, :-1]
        if self.x_flag and self.y_flag:
            return self.l1_loss(pred_grad_x, targ_grad_x) + self.l1_loss(pred_grad_y, targ_grad_y)
        elif self.x_flag:
            return self.l1_loss(pred_grad_x, targ_grad_x)
        elif self.y_flag:
            return self.l1_loss(pred_grad_y, targ_grad_y)
        else:
            raise ValueError('x_flag or y_flag should be True')