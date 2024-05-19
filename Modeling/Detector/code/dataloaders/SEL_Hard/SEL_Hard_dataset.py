import os
import cv2
import math
import pickle
import itertools
import numpy as np
import torch
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataloaders.augmentation_tools import random_rotate_with_line, random_rotate_just_img
from utils.line_utils import find_endpoints, line_equation, transform_theta_to_angle, calculate_distance_from_center
from utils.util import to_np, to_tensor
from utils.misc import coord_norm, coord_unnorm


class Test_Dataset_SEL_Hard(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.max_radius = self.height // 2
        self.max_theta = 90

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])

        datapath = os.path.join(cfg.dataset_root, 'data/SEL_Hard.pickle')
        datalist = pickle.load(open(datapath, 'rb'))

        self.img_list = list()
        self.anno_list = datalist['multiple']
        img_name_list = datalist['img_path']
        self.img_dir = os.path.join(cfg.dataset_root, 'images')
        for idx, img_name in enumerate(img_name_list):
            if os.path.isfile(os.path.join(self.img_dir, f'{img_name}')) is True:
                self.img_list.append(os.path.join(self.img_dir, f'{img_name}'))

        assert len(self.img_list) == len(self.anno_list)

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width),
                                                               interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx, flip=0):
        img = Image.open(os.path.join(self.img_dir, self.img_list[idx])).convert('RGB')
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # transform
        img = self.transform(img)
        return img

    def get_line_pts(self, idx, flip=0):
        img_path = self.img_list[idx]
        annotations = self.anno_list[idx]

        pts_list = list()  # xmin, ymin, xmax, ymax
        for idx in range(len(annotations)):
            annotation = annotations[idx]

            x1 = (float(annotation[0]) - 1) / (400 - 1) * (self.width - 1)
            y1 = (float(annotation[1]) - 1) / (400 - 1) * (self.height - 1)
            x2 = (float(annotation[2]) - 1) / (400 - 1) * (self.width - 1)
            y2 = (float(annotation[3]) - 1) / (400 - 1) * (self.height - 1)
            line = np.array([x1, y1, x2, y2])
            pts = find_endpoints(line.astype('float64'), (self.width-1, self.height-1))

            if pts.tolist() == []:
                continue
            pts_list.append(pts.tolist())

        if flip == 1:
            pts_list = np.float32(pts_list)
            pts_list[:, 0] = self.width - 1 - pts_list[:, 0]
            pts_list[:, 2] = self.width - 1 - pts_list[:, 2]
            return pts_list

        return np.float32(pts_list)

    def __getitem__(self, idx):
        # flip = random.randint(0, 1)
        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img = self.get_image(idx, flip=0)
        gt_lines = self.get_line_pts(idx, flip=0)

        line_eq, check = line_equation(gt_lines)
        theta = transform_theta_to_angle(line_eq)
        radius = calculate_distance_from_center(line_eq, check, gt_lines, self.center)[:, 0, 0]

        need = 6 - len(gt_lines)
        gt_lines = np.concatenate([coord_norm(self.cfg, gt_lines), np.ones((need, 4)) * -1000], axis=0)
        theta = torch.cat([theta, torch.ones(need) * -1000], dim=0)
        radius = torch.cat([radius, torch.ones(need) * -1000], dim=0)
        gt_params = torch.stack((theta, radius), dim=1)

        return {'img_path': img_path,
                'img_name': img_name,
                'img_rgb': img,
                'img': self.normalize(img),
                'gt_lines': gt_lines,
                'gt_params': gt_params}

    def __len__(self):
        return len(self.img_list)
