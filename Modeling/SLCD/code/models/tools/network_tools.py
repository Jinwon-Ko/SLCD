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


# For Combination network
def generate_combinations(n_lines):
    line_idx = np.arange(n_lines)
    comb_list = []
    for i in range(1, 6):
        comb_idx = list(itertools.combinations(line_idx, i))
        for idx in range(len(comb_idx)):
            tmp = np.zeros(n_lines, dtype=bool)
            tmp[list(comb_idx[idx])] = True

            comb_list.append(tmp)
    comb_list = np.array(comb_list)
    return torch.tensor(comb_list, dtype=torch.bool).cuda()


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


class Make_Comb:
    def __init__(self, cfg, scale_factor):
        self.cfg = cfg
        self.sf = scale_factor

        self.topk = cfg.topk

        # generate grid for weight map
        self.grid = {}
        self.X = {}
        self.Y = {}
        self.size = to_tensor(np.float32([self.cfg.width, self.cfg.height, self.cfg.width, self.cfg.height]))
        self.generate_grid(self.cfg.height // self.sf, self.cfg.width // self.sf)

        self.eval_line = Evaluation_Semantic_Line(cfg)
        self.comb_list = self.generate_combinations(cfg.topk)
        print('Num line combinations : ', len(self.comb_list))

    def get_line_pts(self, params):
        init_params = []
        for idx in range(len(params)):
            angle, dist = params[idx]
            endpts = convert_to_line(self.cfg.height, self.cfg.width, angle, dist)
            init_params.append(endpts)
        init_params = torch.FloatTensor(init_params).cuda()
        init_params = init_params.unsqueeze(0)
        return init_params

    def get_line_params(self, line_pts):
        line_pts = to_np(line_pts[0])
        center = np.array([(self.cfg.width - 1) / 2, (self.cfg.height - 1) / 2])
        line_eq, check = line_equation(line_pts)
        angle = transform_theta_to_angle(line_eq)
        dist = calculate_distance_from_center(line_eq, check, line_pts, center)[:, 0, 0]
        angle_dist = torch.stack([angle, dist], dim=1).cuda()
        angle_dist = angle_dist.unsqueeze(0)
        return angle_dist

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

    def generate_dist_map(self, line_eq, sf, check):  # line-point distance
        num = line_eq.shape[0]
        a = line_eq[:, 0].view(num, 1, 1)
        b = line_eq[:, 1].view(num, 1, 1)
        c = line_eq[:, 2].view(num, 1, 1)

        dist = (self.grid[sf][:, :, :, 0] * a + self.grid[sf][:, :, :, 1] * b + c) / torch.sqrt(a * a + b * b)

        if True in check:
            dist[check == True] = (self.grid[sf][:, :, :, 0] - self.line_pts[check == True, 0].view(-1, 1, 1))
        self.dist_map = dist

    def generate_grid(self, height, width):
        X, Y = np.meshgrid(np.linspace(0, width - 1, width),
                           np.linspace(0, height - 1, height))

        self.X[self.sf] = torch.tensor(X, dtype=torch.float, requires_grad=False).cuda()
        self.Y[self.sf] = torch.tensor(Y, dtype=torch.float, requires_grad=False).cuda()
        self.grid[self.sf] = torch.cat((self.X[self.sf].view(1, height, width, 1),
                                        self.Y[self.sf].view(1, height, width, 1)), dim=3)

    def generate_combinations(self, n_lines):
        line_idx = np.arange(n_lines)
        comb_list = []
        for i in range(1, 6):
            comb_idx = list(itertools.combinations(line_idx, i))
            for idx in range(len(comb_idx)):
                tmp = np.zeros(n_lines, dtype=bool)
                tmp[list(comb_idx[idx])] = True

                comb_list.append(tmp)

        return np.array(comb_list)

    def get_maps(self, proposals):
        self.update(proposals, self.sf)
        line_eq, check = self.line_equation()
        self.generate_dist_map(line_eq, self.sf, check)

        line_mask = (-1 <= self.dist_map) * (self.dist_map <= 1)
        region1 = self.dist_map >= 0
        region2 = self.dist_map < 0

        region1 = region1 * 1
        region2 = region2 * -1
        codewords = region1 + region2
        return line_mask, codewords

    def generate_comb_masks(self, line_mask):
        comb_list = self. generate_combinations(len(line_mask))

        line_mask = line_mask.expand(len(comb_list), self.topk, line_mask.shape[1], line_mask.shape[2]).detach().cpu()
        line_mask = line_mask * torch.BoolTensor(comb_list).view(len(comb_list), self.topk, 1, 1)
        combination_mask = line_mask.sum(1)
        combination_mask = combination_mask > 0

        return combination_mask, comb_list   # , position_mask

    def check_results(self, results, gt):
        # Get Topk results
        in_params = results['topk_line_params'].clone().detach()
        gt_params = gt['params']

        in_pts = results['topk_line_pts'].clone().detach()
        gt_pts = gt['pts']

        n_proposal, _ = in_params.shape
        n_gt, _ = gt_params.shape

        ###################################################
        # Calculate Matching Cost
        in_params = in_params.view(n_proposal, 1, 2)
        gt_params = gt_params.view(1, n_gt, 2)

        dist_matrix = (in_params - gt_params)
        dist_matrix[..., 0] /= (self.cfg.max_theta / 2)  # More sensitive on angle displacement
        dist_matrix[..., 1] /= self.cfg.max_radius
        cost_matrix = (dist_matrix ** 2).mean(-1)  # n_proposal, n_gt

        # Check Matching Line
        for gt_idx in range(n_gt):
            matching_idx = torch.topk(cost_matrix[:, gt_idx], 1, largest=False)[1]
            if cost_matrix[matching_idx, gt_idx][0] >= self.cfg.test_nms_threshold:
                gt_line = gt_pts[gt_idx:gt_idx+1]
                in_pts = torch.cat([gt_line, in_pts[:-1]], dim=0)

        results['topk_line_pts'] = in_pts
        return results

    def get_data(self, results, gt):
        in_pts = results['topk_line_pts'].clone().detach()
        gt_pts = gt['pts']

        # k, h, w = codewords.shape
        hiou = []
        score = []
        for i in range(len(self.comb_list)):
            comb_idxes = self.comb_list[i]
            pred_lines = in_pts[comb_idxes]
            # positional_map[i] = to_np(torch.einsum('khw, k -> khw', codewords, torch.tensor(comb_idxes*1).cuda()))
            hiou.append(self.eval_line.measure_HIoU(pred_lines, gt_pts).item())
            score.append(self.eval_line.measure_HIoU_Detector(pred_lines, gt_pts).item())   # Do not consider small piece of separated regions

        hiou = np.array(hiou)
        score = np.array(score)

        return hiou, score

    def save_forwarding_results(self, results, gt, img_name):
        data = dict()

        # Get Topk results
        topk_line_pts = results['topk_line_pts'].float()
        fliped = topk_line_pts.clone()
        fliped[..., 0] = self.cfg.width - 1 - topk_line_pts[..., 0]
        fliped[..., 2] = self.cfg.width - 1 - topk_line_pts[..., 2]
        hious, score = self.get_data(results, gt)

        line_mask, codewords = self.get_maps(topk_line_pts)
        data['0'] = {'topk_line_pts': to_np(topk_line_pts),
                     'codewords': to_np(codewords),
                     'line_mask': to_np(line_mask),
                     'hious': hious,
                     'score': score}

        # Flip version
        line_mask, codewords = self.get_maps(fliped)
        data['1'] = {'topk_line_pts': to_np(fliped),
                     'codewords': to_np(codewords),
                     'line_mask': to_np(line_mask),
                     'hious': hious,
                     'score': score}

        save_name = img_name.replace('jpg', 'pickle')
        save_path = os.path.join(self.cfg.pickle_dir, save_name)
        with open(save_path, mode='wb') as f:
            pickle.dump(data, f)

        match = self.eval_line.measure_PRF(results['topk_line_pts'], gt['pts'])
        return match

    def check_recall(self, results, gt):
        match = self.eval_line.measure_PRF(results['topk_line_pts'], gt['pts'])
        return match

    def save_compositional_feats(self, results, img_name):
        # Get Topk results
        pred_comb = results['pred_comb'].float()
        comp_feat = results['comp_feat'].reshape(-1)

        data = {'pred_comb': to_np(pred_comb),      # [K, 4]
                'comp_feat': to_np(comp_feat)}      # [C, H//4, W//4]

        comb_pickle_dir = '/hdd1/jwko/CVPR2024/Application/Image_Retrieval/pickle/GLD'
        os.makedirs(comb_pickle_dir, exist_ok=True)
        save_name = img_name.replace('jpg', 'pickle')
        save_path = os.path.join(comb_pickle_dir, save_name)
        # save_path = os.path.join(self.cfg.comb_pickle_dir, save_name)
        with open(save_path, mode='wb') as f:
            pickle.dump(data, f)
