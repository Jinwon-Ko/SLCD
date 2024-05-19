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


class Train_Dataset_CDL(Dataset):
    def __init__(self, cfg, flip=True):
        compositions = ['DIAGONAL', 'FRONT', 'HORIZONTAL', 'LOW', 'SYMMETRIC', 'TRIANGLE', 'VERTICAL']

        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.flip = flip

        self.max_radius = self.height // 2
        self.max_theta = 90

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])

        self.img_dir = os.path.join(cfg.dataset_root, 'train/Images')
        self.anno_dir = os.path.join(cfg.dataset_root, 'train/Labels')

        self.img_list = []
        self.anno_list = []
        self.img_class = []
        self.img_compo = []

        # Train mode
        for compo in compositions:
            img_dir = os.path.join(self.img_dir, compo)
            img_list = os.listdir(img_dir)
            for i in range(len(img_list)):
                img_path = os.path.join(img_dir, img_list[i])
                anno_path = os.path.join(img_dir.replace('Images', 'Labels'), img_list[i][:-4] + '.txt')
                with open(anno_path, 'r') as fid:
                    annotations_txt = fid.readlines()
                if len(annotations_txt) == 0:
                    continue

                self.img_list.append(img_path)
                self.anno_list.append(anno_path)
                self.img_compo.append(compo)

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx, flip=0):
        img = cv2.imread(self.img_list[idx])  # [h, w, 3]
        if flip == 1:
            img = cv2.flip(img, flip)

        img = cv2.resize(img, (self.width, self.height))
        return img

    def get_line_pts(self, idx, flip=0):
        img_path = self.img_list[idx]
        anno_path = self.anno_list[idx]

        H, W = cv2.imread(img_path).shape[:2]
        with open(anno_path, 'r') as fid:
            annotations_txt = fid.readlines()

        pts_list = list()  # xmin, ymin, xmax, ymax
        for annotation in annotations_txt:
            annotation_split = annotation.strip().split(' ')

            x1 = float(annotation_split[0]) / W * (self.width - 1)
            y1 = float(annotation_split[1]) / H * (self.height - 1)
            x2 = float(annotation_split[2]) / W * (self.width - 1)
            y2 = float(annotation_split[3]) / H * (self.height - 1)
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

    def get_resized_gt_lines(self, img, gt_line_pts):
        size = [img.shape[1], img.shape[0]]

        gt_lines = np.float32(gt_line_pts)
        if len(gt_lines) != 0:
            gt_lines[:, [0, 2]] = gt_lines[:, [0, 2]] / size[1] * (self.cfg.image_size - 1)
            gt_lines[:, [1, 3]] = gt_lines[:, [1, 3]] / size[0] * (self.cfg.image_size - 1)

        return gt_lines

    def augmentation_random_perpendicular_rotate(self, angle, pts):
        image_center = (self.cfg.image_size / 2, self.cfg.image_size / 2)
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(rot_mat))[0:2, :]

        pts = np.float32(pts)
        if len(pts) != 0:
            src_pt1 = pts[:, [0, 1]]
            src_pt2 = pts[:, [2, 3]]
            pts[:, [0, 1]] = src_pt1 * rot_mat_notranslate
            pts[:, [2, 3]] = src_pt2 * rot_mat_notranslate

            ones = np.ones(shape=(len(src_pt1), 1))     # [N, 1]
            pt1_ones = np.hstack([src_pt1, ones])       # [N, 3]
            dst_pt1 = affine_mat.dot(pt1_ones.T).T

            ones = np.ones(shape=(len(src_pt2), 1))     # [N, 1]
            pt2_ones = np.hstack([src_pt2, ones])       # [N, 3]
            dst_pt2 = affine_mat.dot(pt2_ones.T).T

            pts[:, [0, 1]] = dst_pt1
            pts[:, [2, 3]] = dst_pt2

        return pts

    def __getitem__(self, idx):
        seed = random.randint(0, 2 ** 32)
        np.random.seed(seed)
        flip = random.randint(0, 1) if self.flip else 0

        img_path = self.img_list[idx]
        img_name = os.path.split(img_path)[1]
        img = self.get_image(idx, flip=flip)    # cv2 image
        gt_lines = self.get_line_pts(idx, flip=flip)

        # random rotate
        if self.cfg.perpendicular_rotate:
            perpendicular_rotate = random.randint(0, 3) * 90
            img = random_rotate_just_img(img, perpendicular_rotate)
            gt_lines = self.augmentation_random_perpendicular_rotate(perpendicular_rotate, gt_lines)

        if self.cfg.random_rotate:
            rotate_angle = np.random.uniform(-5, 5)
            img, gt_lines = random_rotate_with_line(self.cfg, img, rotate_angle, gt_lines)
            gt_lines = self.get_resized_gt_lines(img, gt_lines)

        # transform
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)

        line_eq, check = line_equation(gt_lines)
        theta = transform_theta_to_angle(line_eq)
        radius = calculate_distance_from_center(line_eq, check, gt_lines, self.center)[:, 0, 0]

        need = 10 - len(gt_lines)
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


class Test_Dataset_CDL(Dataset):
    def __init__(self, cfg):
        compositions = ['DIAGONAL', 'FRONT', 'HORIZONTAL', 'LOW', 'SYMMETRIC', 'TRIANGLE', 'VERTICAL']

        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.max_radius = self.height // 2
        self.max_theta = 90

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])

        self.img_dir = os.path.join(cfg.dataset_root, 'test/Images')
        self.anno_dir = os.path.join(cfg.dataset_root, 'test/Labels')

        self.img_list = []
        self.anno_list = []
        self.img_class = []
        self.img_compo = []

        # Test mode
        for compo in compositions:
            img_dir = os.path.join(self.img_dir, compo)
            img_list = os.listdir(img_dir)
            for i in range(len(img_list)):
                img_path = os.path.join(img_dir, img_list[i])
                anno_path = os.path.join(img_dir.replace('Images', 'Labels'), img_list[i][:-4] + '.txt')
                with open(anno_path, 'r') as fid:
                    annotations_txt = fid.readlines()
                if len(annotations_txt) == 0:
                    continue

                self.img_list.append(img_path)
                self.anno_list.append(anno_path)
                self.img_compo.append(compo)

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx, flip=0):
        img = Image.open(self.img_list[idx]).convert('RGB')
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # transform
        img = self.transform(img)
        return img

    def get_line_pts(self, idx, flip=0):
        img_path = self.img_list[idx]
        anno_path = self.anno_list[idx]

        H, W = cv2.imread(img_path).shape[:2]
        with open(anno_path, 'r') as fid:
            annotations_txt = fid.readlines()

        pts_list = list()  # xmin, ymin, xmax, ymax
        for annotation in annotations_txt:
            annotation_split = annotation.strip().split(' ')

            x1 = float(annotation_split[0]) / W * (self.width - 1)
            y1 = float(annotation_split[1]) / H * (self.height - 1)
            x2 = float(annotation_split[2]) / W * (self.width - 1)
            y2 = float(annotation_split[3]) / H * (self.height - 1)
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

        need = 10 - len(gt_lines)
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
