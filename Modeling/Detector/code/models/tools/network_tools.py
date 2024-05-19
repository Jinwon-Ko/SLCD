import os
import torch
import numpy as np
import pickle
import itertools

from utils.util import to_np, to_tensor
from utils.line_utils import get_line_pts_from_params, get_line_pts_from_normed_params, convert_to_line, \
    line_equation, transform_theta_to_angle, calculate_distance_from_center
from utils.calculate_metrics import Evaluation_Semantic_Line

class Make_features:
    def __init__(self, cfg):
        self.cfg = cfg

        # generate grid for weight map
        self.size = to_tensor(np.float32([self.cfg.width, self.cfg.height, self.cfg.width, self.cfg.height]))

    def generate_grid(self, height, width):
        X, Y = np.meshgrid(np.linspace(0, width - 1, width),
                           np.linspace(0, height - 1, height))

        self.X = torch.tensor(X, dtype=torch.float, requires_grad=False).cuda()
        self.Y = torch.tensor(Y, dtype=torch.float, requires_grad=False).cuda()
        self.grid = torch.cat((self.X.view(1, height, width, 1),
                               self.Y.view(1, height, width, 1)), dim=3)

    def update(self, line_pts, scale_factor):
        self.line_pts = line_pts / (self.size - 1) * (self.size // scale_factor - 1)

    def line_equation(self):
        data = self.line_pts.clone()

        # data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
        line_eq = torch.zeros((data.shape[0], 3)).cuda()
        line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
        line_eq[:, 1] = -1
        line_eq[:, 2] = -1 * line_eq[:, 0] * data[:, 0] + data[:, 1]
        check = ((data[:, 0] - data[:, 2]) == 0)

        return line_eq, check

    def generate_dist_map(self, line_eq, check):  # line-point distance
        num = line_eq.shape[0]
        a = line_eq[:, 0].view(num, 1, 1)
        b = line_eq[:, 1].view(num, 1, 1)
        c = line_eq[:, 2].view(num, 1, 1)

        dist = (self.grid[:, :, :, 0] * a + self.grid[:, :, :, 1] * b + c) / torch.sqrt(a * a + b * b)

        if True in check:
            dist[check == True] = (self.grid[:, :, :, 0] - self.line_pts[check == True, 0].view(-1, 1, 1))
        self.dist_map = dist

    def generate_gaussian_dist_map(self, params, sf):
        self.generate_grid(self.cfg.height // sf, self.cfg.width // sf)
        proposals = get_line_pts_from_params(self.cfg, params, return_norm=False)
        self.update(proposals, sf)
        line_eq, check = self.line_equation()
        self.generate_dist_map(line_eq, check)

        dist_map = torch.abs(self.dist_map)
        weight_map = torch.exp(-1 * torch.pow(dist_map, 2) / (2 * self.cfg.gaussian_sigma))
        return weight_map       # [N, h, w]

    def get_line_masks(self, params, sf, margin=0.5):
        self.generate_grid(self.cfg.height // sf, self.cfg.width // sf)
        proposals = get_line_pts_from_normed_params(self.cfg, params, return_norm=False)
        self.update(proposals, sf)
        line_eq, check = self.line_equation()
        self.generate_dist_map(line_eq, check)

        line_mask = (-margin <= self.dist_map) * (self.dist_map <= margin)
        n_l = torch.sum(line_mask, dim=[-2, -1]) + 1e-4
        return line_mask, n_l    # [N, h, w], [N]

    def get_region_masks(self, params, sf):
        self.generate_grid(self.cfg.height // sf, self.cfg.width // sf)
        proposals = get_line_pts_from_params(self.cfg, params, return_norm=False)
        self.update(proposals, sf)
        line_eq, check = self.line_equation()
        self.generate_dist_map(line_eq, check)

        mask_r = (self.dist_map >= 0) * 1 + (self.dist_map < 0) * -1
        return mask_r    # [N, h, w]

    def get_seperate_region_masks(self, params, sf):
        self.generate_grid(self.cfg.height // sf, self.cfg.width // sf)
        proposals = get_line_pts_from_params(self.cfg, params, return_norm=False)
        self.update(proposals, sf)
        line_eq, check = self.line_equation()
        self.generate_dist_map(line_eq, check)

        mask_r = (self.dist_map >= 0) * 1 + (self.dist_map < 0) * -1
        dist_map = torch.abs(self.dist_map)
        weight_map = torch.exp(-1 * torch.pow(dist_map, 2) / (2 * self.cfg.gaussian_sigma))
        # mask_r1 = weight_map * (self.dist_map >= 0)
        # mask_r2 = weight_map * (self.dist_map < 0)
        # n_r1 = torch.sum(self.dist_map >= 0, dim=[1, 2])
        # n_r2 = torch.sum(self.dist_map < 0, dim=[1, 2])
        return mask_r, weight_map

    def get_line_region_masks(self, params, sf, margin=1):
        self.generate_grid(self.cfg.height // sf, self.cfg.width // sf)
        proposals = get_line_pts_from_params(self.cfg, params, return_norm=False)
        self.update(proposals, sf)
        line_eq, check = self.line_equation()
        self.generate_dist_map(line_eq, check)

        dist_map = torch.abs(self.dist_map)
        line_mask = (-margin <= self.dist_map) * (self.dist_map <= margin)
        region_mask1 = self.dist_map >= margin
        region_mask2 = self.dist_map < -margin
        weight_map = torch.exp(-1 * torch.pow(dist_map, 2) / (2 * self.cfg.gaussian_sigma))

        n_l = torch.sum(line_mask, dim=[1, 2]) + 1e-4
        n_r1 = torch.sum(region_mask1, dim=[1, 2]) + 1e-4
        n_r2 = torch.sum(region_mask2, dim=[1, 2]) + 1e-4
        return


class Post_Process:
    def __init__(self, cfg):
        self.cfg = cfg

    def detect_topk(self, line_pts, probs):
        select_idx = []
        for batch_idx in range(len(probs)):
            prob = to_np(probs[batch_idx])
            proposals = to_np(line_pts[batch_idx])
            center = np.array([(self.cfg.width - 1) / 2, (self.cfg.height - 1) / 2])

            sorted_idx = np.argsort(prob)[::-1]

            line_eq, check = line_equation(proposals)
            angle = transform_theta_to_angle(line_eq)
            dist = calculate_distance_from_center(line_eq, check, proposals, center)[:, 0, 0]

            angle_dist = torch.stack([angle, dist], dim=1)
            angle_dist[..., 0] /= (self.cfg.max_theta / 2)     # More sensitive on angle displacement
            angle_dist[..., 1] /= self.cfg.max_radius
            visit_mask = torch.ones(len(proposals), dtype=torch.int64)
            count = 0

            assert len(sorted_idx) == len(angle_dist)

            for idx in range(len(angle_dist)):
                now = sorted_idx[idx]
                if count == self.cfg.topk:
                    break
                if visit_mask[now] == 0:
                    continue
                dist = angle_dist - angle_dist[now]
                cost = (dist ** 2).mean(-1)
                cost[now] = 10000
                delete_idx = torch.where(cost < self.cfg.test_nms_threshold)[0]
                visit_mask[delete_idx] = 0
                visit_mask[now] = 0

                count += 1
                select_idx.append(now)

        select_idx = torch.LongTensor(select_idx)
        select_idx = select_idx.reshape(len(probs), self.cfg.topk)
        return select_idx
