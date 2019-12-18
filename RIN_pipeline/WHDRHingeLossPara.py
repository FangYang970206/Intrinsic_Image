import torch
import torch.nn as nn
from torch.autograd import Function
import math


class WHDRHingeLossPara(Function):
    # def __init__(self, delta, epsilon, device):
    #     0.12 = delta
    #     0.08 = epsilon
    #     self.device = device
    
    @staticmethod
    def forward(self, inp, target):
        self.inp = inp
        self.target = target
        _, _, height, width = inp.size()
        print(inp.size())
        self.whdr = 0
        self.weight = 0
        for i in range(target.size()[1]):
            self.weight += self.target[:, i, 0]
            point1_x = math.floor(width * self.target[:, i, 2])
            point1_y = math.floor(height * self.target[:, i, 3])
            point2_x = math.floor(width * self.target[:, i, 4])
            point2_y = math.floor(height * self.target[:, i, 5])
            ratio = inp[0][0][point1_y][point1_x] / (inp[0][0][point2_y][point2_x] + 1e-7)
            predict_j = -1
            if ratio > (1 + 0.12):
                predict_j = 2
            elif ratio < 1/(1 + 0.12):
                predict_j = 1
            else:
                predict_j = 0
            
            if int(self.target[:, i, 1]) != predict_j:
                self.whdr += self.target[:, i, 0]
        
        self.whdr = self.whdr / self.weight
        return self.whdr

    @staticmethod
    def backward(self, grad_output):
        _, _, height, width = self.inp.size()
        self.grad_input = torch.zeros_like(self.inp).to(self.inp.device)
        for i in range(self.target.size()[1]):
            point1_x = math.floor(width * self.target[:, i, 2])
            point1_y = math.floor(height * self.target[:, i, 3])
            point2_x = math.floor(width * self.target[:, i, 4])
            point2_y = math.floor(height * self.target[:, i, 5])
            ratio = self.inp[0][0][point1_y][point1_x] / (self.inp[0][0][point2_y][point2_x] + 1e-7)
            if int(self.target[:, i, 1]) == 0:
                if ratio < 1 / (1 + 0.12 - 0.08):
                    self.grad_input[:, :, point1_y, point1_x] -= 1 / (self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
                    self.grad_input[:, :, point2_y, point2_x] += self.inp[0][0][point1_y][point1_x] / (self.inp[0][0][point2_y][point2_x] * self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
                elif ratio > (1 + 0.12 - 0.08):
                    self.grad_input[:, :, point1_y, point1_x] += 1 / (self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
                    self.grad_input[:, :, point2_y, point2_x] -= self.inp[0][0][point1_y][point1_x] / (self.inp[0][0][point2_y][point2_x] * self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
            elif int(self.target[:, i, 1]) == 1:
                if ratio > 1 / (1 + 0.12 + 0.08):
                    self.grad_input[:, :, point1_y, point1_x] += 1 / (self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
                    self.grad_input[:, :, point2_y, point2_x] -= self.inp[0][0][point1_y][point1_x] / (self.inp[0][0][point2_y][point2_x] * self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
            else:
                if ratio < (1 + 0.12 + 0.08):
                    self.grad_input[:, :, point1_y, point1_x] -= 1 / (self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
                    self.grad_input[:, :, point2_y, point2_x] += self.inp[0][0][point1_y][point1_x] / (self.inp[0][0][point2_y][point2_x] * self.inp[0][0][point2_y][point2_x] + 1e-7) * self.target[:, i, 0]
        self.grad_input = self.grad_input / self.weight
        return self.grad_input, None


class WHDRHingeLossParaModule(nn.Module):

    def forward(self, input, target):
        return WHDRHingeLossPara.apply(input, target)
