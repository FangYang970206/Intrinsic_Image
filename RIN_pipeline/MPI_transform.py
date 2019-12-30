import torchvision.transforms as transforms
from PIL import Image
import random, math
import numpy as np
import torch


class MPI_Test_Agumentation(object):
    '''MPI test dataset prepocessing'''
    def __init__(self):
        self.toTensor = transforms.ToTensor()
    def __call__(self, input_image, albedo_image, shading_image, mask):
        return self.toTensor(input_image),self.toTensor(albedo_image),self.toTensor(shading_image), self.toTensor(mask)


class MPI_Train_Agumentation(object):
    '''MPI naive training dataset prepocessing'''
    def __init__(self, size=256, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, input_image, albedo_image, shading_image, mask=None):
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random crop
        area = im_w * im_h
        target_area, aspect_ratio = random.uniform(0.2, 0.8)*area, random.uniform(3./4, 4./3)
        tmp_w, tmp_h = int(round(math.sqrt(target_area * aspect_ratio))), int(round(math.sqrt(target_area / aspect_ratio)))
        tmp_w, tmp_h = (tmp_h, tmp_w) if random.random() < 0.5 else (tmp_w, tmp_h)
        if tmp_w <= im_w and tmp_h <= im_h:
            start_x, start_y = random.randint(0, im_w-tmp_w), random.randint(0, im_h-tmp_h)
            input_image = input_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            albedo_image = albedo_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            shading_image = shading_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random scale to [0.8 - 1.2]
        scale_to_size = int(random.uniform(0.8, 1.2) * min(im_w, im_h))
        clc_scale = transforms.Resize(scale_to_size)
        input_image, albedo_image, shading_image = clc_scale(input_image), clc_scale(albedo_image), clc_scale(shading_image)
        # random left-right flip  with probability 0.5
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            albedo_image = albedo_image.transpose(Image.FLIP_LEFT_RIGHT)
            shading_image = shading_image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
            albedo_image = albedo_image.transpose(Image.FLIP_TOP_BOTTOM)
            shading_image = shading_image.transpose(Image.FLIP_TOP_BOTTOM)

        input_image = input_image.resize((self.size, self.size), self.interpolation)
        albedo_image = albedo_image.resize((self.size, self.size), self.interpolation)
        shading_image = shading_image.resize((self.size, self.size), self.interpolation)

        # albedo_image, shading_image = np.array(albedo_image), np.array(shading_image)
        albedo_image, shading_image = np.array(albedo_image), np.array(shading_image)
        mask = np.repeat((albedo_image.mean(2) != 0).astype(np.uint8)[..., np.newaxis]*255, 3, 2)
        input_image = (albedo_image.astype(np.float32)/255)*(shading_image.astype(np.float32)/255)*255
        # print(shading_image.shape)
        albedo_image = (torch.from_numpy(albedo_image.transpose((2, 0, 1)))).float().div(255)
        shading_image = (torch.from_numpy(shading_image.transpose((2, 0, 1)))).float().div(255)
        mask = (torch.from_numpy(mask.transpose((2, 0, 1)))).float().div(255)
        input_image = (torch.from_numpy(input_image.transpose((2, 0, 1)))).float().div(255)
        # print(input_image.size())
        return input_image, albedo_image, shading_image, mask

class MPI_Train_Agumentation_fy(object):
    '''MPI naive training dataset prepocessing'''
    def __init__(self, size=256, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, input_image, albedo_image, shading_image, mask=None):
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random crop
        area = im_w * im_h
        target_area, aspect_ratio = random.uniform(0.2, 0.8)*area, random.uniform(3./4, 4./3)
        tmp_w, tmp_h = int(round(math.sqrt(target_area * aspect_ratio))), int(round(math.sqrt(target_area / aspect_ratio)))
        tmp_w, tmp_h = (tmp_h, tmp_w) if random.random() < 0.5 else (tmp_w, tmp_h)
        if tmp_w <= im_w and tmp_h <= im_h:
            start_x, start_y = random.randint(0, im_w-tmp_w), random.randint(0, im_h-tmp_h)
            input_image = input_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            albedo_image = albedo_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            shading_image = shading_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random scale to [0.8 - 1.2]
        scale_to_size = int(random.uniform(0.8, 1.2) * min(im_w, im_h))
        clc_scale = transforms.Resize(scale_to_size)
        input_image, albedo_image, shading_image = clc_scale(input_image), clc_scale(albedo_image), clc_scale(shading_image)
        # random left-right flip  with probability 0.5
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            albedo_image = albedo_image.transpose(Image.FLIP_LEFT_RIGHT)
            shading_image = shading_image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
            albedo_image = albedo_image.transpose(Image.FLIP_TOP_BOTTOM)
            shading_image = shading_image.transpose(Image.FLIP_TOP_BOTTOM)

        input_image = input_image.resize((self.size, self.size), self.interpolation)
        albedo_image = albedo_image.resize((self.size, self.size), self.interpolation)
        shading_image = shading_image.resize((self.size, self.size), self.interpolation)

        # albedo_image, shading_image = np.array(albedo_image), np.array(shading_image)
        # albedo_image, shading_image = np.array(albedo_image), np.array(shading_image)
        # mask = np.repeat((albedo_image.mean(2) != 0).astype(np.uint8)[..., np.newaxis]*255, 3, 2)
        # input_image = (albedo_image.astype(np.float32)/255)*(shading_image.astype(np.float32)/255)*255
        # print(shading_image.shape)
        # albedo_image = (torch.from_numpy(albedo_image.transpose((2, 0, 1)))).float().div(255)
        # shading_image = (torch.from_numpy(shading_image.transpose((2, 0, 1)))).float().div(255)
        # mask = (torch.from_numpy(mask.transpose((2, 0, 1)))).float().div(255)
        # input_image = (torch.from_numpy(input_image.transpose((2, 0, 1)))).float().div(255)
        # print(input_image.size())
        return input_image, albedo_image, shading_image


class MPI_Train_Agumentation_fy2(object):
    '''MPI naive training dataset prepocessing'''
    def __init__(self, size=256, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, input_image, albedo_image, shading_image, mask=None):
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random crop
        area = im_w * im_h
        target_area, aspect_ratio = random.uniform(0.5, 0.8)*area, random.uniform(3./4, 4./3)
        tmp_w, tmp_h = int(round(math.sqrt(target_area * aspect_ratio))), int(round(math.sqrt(target_area / aspect_ratio)))
        tmp_w, tmp_h = (tmp_h, tmp_w) if random.random() < 0.5 else (tmp_w, tmp_h)
        if tmp_w <= im_w and tmp_h <= im_h:
            start_x, start_y = random.randint(0, im_w-tmp_w), random.randint(0, im_h-tmp_h)
            input_image = input_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            albedo_image = albedo_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
            shading_image = shading_image.crop((start_x, start_y, start_x+tmp_w, start_y+tmp_h))
        im_w, im_h = input_image.size[0], input_image.size[1]
        # random scale to [0.8 - 1.2]
        scale_to_size = int(random.uniform(0.8, 1.2) * min(im_w, im_h))
        clc_scale = transforms.Resize(scale_to_size)
        input_image, albedo_image, shading_image = clc_scale(input_image), clc_scale(albedo_image), clc_scale(shading_image)
        # random left-right flip  with probability 0.5
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            albedo_image = albedo_image.transpose(Image.FLIP_LEFT_RIGHT)
            shading_image = shading_image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
            albedo_image = albedo_image.transpose(Image.FLIP_TOP_BOTTOM)
            shading_image = shading_image.transpose(Image.FLIP_TOP_BOTTOM)

        input_image = input_image.resize((self.size, self.size), self.interpolation)
        albedo_image = albedo_image.resize((self.size, self.size), self.interpolation)
        shading_image = shading_image.resize((self.size, self.size), self.interpolation)

        return input_image, albedo_image, shading_image