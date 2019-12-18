import torch
import torch.nn as nn
from torch.autograd import Function


class WHDRHingeLossParaPro(Function):
    
    @staticmethod
    def forward(self, inp, target):
        _, _, height, width = inp.size()
        #print("inp size: ", inp.size())
        #print("target size: ", target.size())
        weight = torch.sum(target[0, :, 0])
        point1_x = torch.floor(width * target[0, :, 2]).long()
        #print("point1_x size: ", point1_x.size())
        point1_y = torch.floor(height * target[0, :, 3]).long()
        #print("point1_y size: ", point1_y.size())
        point2_x = torch.floor(width * target[0, :, 4]).long()
        #print("point2_x size: ", point2_x.size())
        point2_y = torch.floor(height * target[0, :, 5]).long()
        #print("point2_y size: ", point2_y.size())
        divisor = torch.index_select(torch.index_select(inp[0][0], 0, point1_y), 1, point1_x)
        #print("divisor size: ", divisor.size())
        divisor = torch.masked_select(divisor, torch.eye(divisor.size()[0]).ge(0.5).to(inp.device))
        #print("divisor size: ", divisor.size())
        dividend = torch.index_select(torch.index_select(inp[0][0], 0, point2_y), 1, point2_x)
        #print("dividend size: ", dividend.size())
        dividend = torch.masked_select(dividend, torch.eye(dividend.size()[0]).ge(0.5).to(inp.device))
        #print("dividend size: ", dividend.size())
        ratio = divisor / (dividend + 1e-7)
        #print("ratio size: ", ratio.size())
        self.save_for_backward(inp, target, weight, point1_x, point1_y, point2_x, point2_y, ratio)
        predict = ratio.where(ratio <= (1 + 0.12), torch.ones(ratio.size()).to(inp.device) * 2)
        #print("predict size: ", predict.size())
        predict  = predict.where(predict >= 1/(1 + 0.12), torch.ones(predict.size()).to(inp.device))
        #print("predict size: ", predict.size())
        predict = predict.where(predict == 1, torch.zeros(predict.size()).to(inp.device)) + \
                predict.where(predict == 2, torch.zeros(predict.size()).to(inp.device))
        #print("predict size: ", predict.size())
        #print(predict)
        whdr = torch.sum(torch.where(target[0, :, 1] != predict, target[0, :, 0], torch.zeros(target[0, :, 0].size()).to(inp.device)))
        # whdr = torch.sum(target[0, :, 1].where(target[0, :, 1] != predict, torch.zeros(target[0, :, 1].size()).to(inp.device)))
        #print(whdr)
        return whdr / weight

    @staticmethod
    def backward(self, grad_output):
        inp, target, weight, point1_x, point1_y, point2_x, point2_y, ratio = self.saved_tensors
        # _, _, height, width = inp.size()
        grad_input = torch.zeros_like(inp).to(inp.device)
        # mask1 = (target[0, :, 1]==0) * (ratio < 1 / (1 + 0.12 - 0.08))
        # point1_y_1 = torch.masked_select(point1_y, mask1)
        # point1_x_1 = torch.masked_select(point1_x, mask1)
        # point2_y_1 = torch.masked_select(point2_y, mask1)
        # point2_x_1 = torch.masked_select(point2_x, mask1)
        # target_weight_1 = torch.masked_select(target[0, :, 0], mask1)

        # inp_y1_x1 = torch.index_select(torch.index_select(inp[0][0], 0, point1_y_1), 1, point1_x_1)
        # inp_y1_x1 = torch.masked_select(inp_y1_x1, torch.eye(inp_y1_x1.size()[0]).ge(0.5))
        
        # inp_y2_x2 = torch.index_select(torch.index_select(inp[0][0], 0, point2_y_1), 1, point2_x_1)
        # inp_y2_x2 = torch.masked_select(inp_y2_x2, torch.eye(inp_y2_x2.size()[0]).ge(0.5))
        
        # grad1 = torch.index_select(torch.index_select(grad_input[0][0], 2, point1_y_1), 3, point1_x_1)
        # grad1 = torch.masked_select(grad1, torch.eye(grad1.size()[0]).ge(0.5))
        target_weight = target[0, :, 0]
        target_label = target[0, :, 1].long()
        for i in range(target_label.size()[0]):
            if target_label[i] == 0:
                if ratio[i] < 1 / (1 + 0.12 - 0.08):
                    grad_input[:, :, point1_y[i], point1_x[i]] -= 1 / (inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
                    grad_input[:, :, point2_y[i], point2_x[i]] += inp[0][0][point1_y[i]][point1_x[i]] / (inp[0][0][point2_y[i]][point2_x[i]] * inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
                elif ratio[i] > (1 + 0.12 - 0.08):
                    grad_input[:, :, point1_y[i], point1_x[i]] += 1 / (inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
                    grad_input[:, :, point2_y[i], point2_x[i]] -= inp[0][0][point1_y[i]][point1_x[i]] / (inp[0][0][point2_y[i]][point2_x[i]] * inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
            elif target_label[i] == 1:
                if ratio[i] > 1 / (1 + 0.12 + 0.08):
                    grad_input[:, :, point1_y[i], point1_x[i]] += 1 / (inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
                    grad_input[:, :, point2_y[i], point2_x[i]] -= inp[0][0][point1_y[i]][point1_x[i]] / (inp[0][0][point2_y[i]][point2_x[i]] * inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
            else:
                if ratio[i] < (1 + 0.12 + 0.08):
                    grad_input[:, :, point1_y[i], point1_x[i]] -= 1 / (inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
                    grad_input[:, :, point2_y[i], point2_x[i]] += inp[0][0][point1_y[i]][point1_x[i]] / (inp[0][0][point2_y[i]][point2_x[i]] * inp[0][0][point2_y[i]][point2_x[i]] + 1e-7) * target_weight[i]
        grad_input = grad_input / weight
        return grad_input, None


class WHDRHingeLossParaProModule(nn.Module):

    def forward(self, input, target):
        return WHDRHingeLossParaPro.apply(input, target)