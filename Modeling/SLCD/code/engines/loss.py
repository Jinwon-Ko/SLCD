import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy.optimize import linear_sum_assignment

from utils.misc import accuracy, sigmoid_focal_loss
from utils.util import *


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll1 = nn.NLLLoss(reduce=True)
        self.nll2 = nn.NLLLoss(reduce=False)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=-1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=-1)
        log_score = factor * log_score

        loss = self.nll1(log_score.transpose(1, 2), labels)

        return loss

class Loss_Function(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth lines and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and line)
    """
    def __init__(self, cfg, weight_dict):

        super().__init__()
        self.cfg = cfg
        self.num_classes = 1
        self.weight_dict = weight_dict
        self.focal_alpha = cfg.focal_alpha

        self.loss_focal = SoftmaxFocalLoss(gamma=2)
        self.loss_KLD = nn.KLDivLoss(reduction='batchmean', log_target=True)

    # Losses
    def get_KLDloss_each_pairs(self, outputs, targets, eps=1e-6):
        if not self.cfg.use_KLD:
            return {'KLD': torch.tensor(0)}
        loss_kld = []
        sizes = [len(torch.nonzero(v[:, 0] != -1000)) for v in targets['params']]
        for batch_idx in range(len(sizes)):
            loss_kld_b = []

            src_probs = outputs[f'attn_map']            # [L, b, N, h*w]
            tgt_probs = outputs[f'positional_map']      # [b, M, h, w]
            for line_idx in range(sizes[batch_idx]):
                region_map = tgt_probs[batch_idx, line_idx]
                region_map_flatten = region_map.flatten()

                group1 = torch.where(region_map_flatten == 1.)[0]
                group2 = torch.where(region_map_flatten == -1.)[0]

                group1_prob = src_probs[:, batch_idx, :, group1].mean(-1)
                group2_prob = src_probs[:, batch_idx, :, group2].mean(-1)

                loss_kld_b += [-self.loss_KLD((group1_prob + eps).log(), (group2_prob + eps).log())]  # src & tgt must log_softmax
                loss_kld_b += [-self.loss_KLD((group2_prob + eps).log(), (group1_prob + eps).log())]

            loss_kld += [torch.stack(loss_kld_b).mean()]

        loss_kld = torch.stack(loss_kld)
        return {'KLD': loss_kld.mean() * self.weight_dict['loss_kld']}

    def get_loss_for_line_detection(self, outputs, targets):
        losses = {'loss_cls': 0, 'loss_reg': 0}
        loss_cls, loss_reg = self.loss_for_nms_version(outputs, targets)
        losses[f'loss_cls'] += self.weight_dict['loss_cls'] * loss_cls
        losses[f'loss_reg'] += self.weight_dict['loss_reg'] * loss_reg
        return losses

    def loss_for_nms_version(self, outputs, targets):
        loss_reg = 0
        loss_cls = 0

        sizes = [len(torch.nonzero(v[:, 0] != -1000)) for v in targets['pts']]
        valid = targets['pts'][..., 0] != -1000
        pred_logit = outputs[f'cls']
        gt_prob = torch.zeros(outputs[f'params'].shape[:-1], dtype=torch.int64).cuda()

        init_params = outputs[f'params_init'].clone()       # [b, N, 2]
        src_params = outputs[f'params'].clone()             # [b, N, 2]
        tgt_params = targets['params'].clone()              # [b, T, 2]

        tgt_params[..., 0] = (tgt_params[..., 0] + self.cfg.max_theta) / (2 * self.cfg.max_theta)
        tgt_params[..., 1] = (tgt_params[..., 1] + self.cfg.max_radius) / (2 * self.cfg.max_radius)

        for b_idx, n_gt in enumerate(sizes):
            valid_b = valid[b_idx]
            init_params_b = init_params[b_idx]              # [N, 2]
            src_params_b = src_params[b_idx]                # [N, 2]
            tgt_params_b = tgt_params[b_idx][valid_b]       # [T, 2]

            # set positive lines
            theta_diff = torch.abs(tgt_params_b[..., 0].unsqueeze(0) - init_params_b[..., 0].unsqueeze(1))
            radius_diff = torch.abs(tgt_params_b[..., 1].unsqueeze(0) - init_params_b[..., 1].unsqueeze(1))

            for gt_idx in range(n_gt):
                condition1 = theta_diff[:, gt_idx] < (self.cfg.threshold_theta / self.cfg.max_theta)
                condition2 = radius_diff[:, gt_idx] < (self.cfg.threshold_radius / self.cfg.max_radius)
                condition = torch.logical_and(condition1, condition2)
                pos_lines = torch.where(condition)[0]
                if len(pos_lines) == 0:
                    pos_lines = torch.argmin(radius_diff[:, gt_idx] ** 2 + theta_diff[:, gt_idx] ** 2, keepdim=True)
                gt_prob[b_idx, pos_lines] = 1

                # cost using params
                dist = tgt_params_b[gt_idx] - src_params_b[pos_lines]       # [len(pos_lines), 2]
                cost = (dist ** 2).mean(-1)
                loss_reg += cost.sum()

        loss_reg = loss_reg / len(sizes)
        loss_cls += self.loss_focal(pred_logit, gt_prob)
        return loss_cls, loss_reg

    def loss_for_nms_version_backup(self, outputs, targets):
        loss_reg = 0
        loss_cls = 0

        sizes = [len(torch.nonzero(v[:, 0] != -1000)) for v in targets['pts']]
        valid = targets['pts'][..., 0] != -1000
        pred_logit = outputs[f'cls']
        gt_prob = torch.zeros(outputs[f'params'].shape[:-1], dtype=torch.int64).cuda()

        src_params = outputs[f'params_init'].clone()        # [b, N, 2]
        tgt_params = targets['params'].clone()              # [b, T, 2]
        src_pts = outputs[f'pts']                           # [b, N, 4]
        tgt_pts = targets['pts']                            # [b, T, 4]

        pos_line_list = []
        for b_idx, n_gt in enumerate(sizes):
            valid_b = valid[b_idx]
            src_params_b = src_params[b_idx]
            tgt_params_b = tgt_params[b_idx][valid_b]
            src_pts_b = src_pts[b_idx]
            tgt_pts_b = tgt_pts[b_idx][valid_b]

            # cost using params
            src_params_b[..., 0] = src_params_b[..., 0] / (self.cfg.max_theta / 2)  # More sensitive on angle displacement
            tgt_params_b[..., 0] = tgt_params_b[..., 0] / (self.cfg.max_theta / 2)  # More sensitive on angle displacement
            src_params_b[..., 1] = src_params_b[..., 1] / self.cfg.max_radius
            tgt_params_b[..., 1] = tgt_params_b[..., 1] / self.cfg.max_radius

            cost_matrix_b = ((tgt_params_b.unsqueeze(0) - src_params_b.unsqueeze(1)) ** 2).mean(-1)     # [n_query, n_gt]
            for gt_idx in range(n_gt):
                cost_vector = cost_matrix_b[:, gt_idx]
                pos_lines = torch.where(cost_vector < self.cfg.loss_nms_threshold)[0]
                if len(pos_lines) == 0:
                    pos_lines = torch.argmin(cost_vector, keepdim=True)
                gt_prob[b_idx, pos_lines] = 1

                # cost using pts
                dist1 = tgt_pts_b[gt_idx] - src_pts_b[pos_lines]                      # [len(pos_lines), 4]
                dist2 = tgt_pts_b[gt_idx][..., [2, 3, 0, 1]] - src_pts_b[pos_lines]   # [len(pos_lines), 4]
                cost1 = (dist1 ** 2).mean(-1)
                cost2 = (dist2 ** 2).mean(-1)
                cost = torch.gt(cost1, cost2) * cost2 + torch.le(cost1, cost2) * cost1

                loss_reg += cost.sum()

        loss_reg = loss_reg / len(sizes)
        loss_cls += self.loss_focal(pred_logit, gt_prob)
        return loss_cls, loss_reg

    def get_pos_lines(self, outputs, targets):
        sizes = [len(torch.nonzero(v[:, 0] != -1000)) for v in targets['pts']]
        valid = targets['pts'][..., 0] != -1000

        init_params = outputs[f'params_init'].clone()       # [b, N, 2]
        tgt_params = targets['params'].clone()              # [b, T, 2]

        tgt_params[..., 0] = (tgt_params[..., 0] + self.cfg.max_theta) / (2 * self.cfg.max_theta)
        tgt_params[..., 1] = (tgt_params[..., 1] + self.cfg.max_radius) / (2 * self.cfg.max_radius)

        pos_line_list = []
        for b_idx, n_gt in enumerate(sizes):
            valid_b = valid[b_idx]
            init_params_b = init_params[b_idx]              # [N, 2]
            tgt_params_b = tgt_params[b_idx][valid_b]       # [T, 2]

            # set positive lines
            theta_diff = torch.abs(tgt_params_b[..., 0].unsqueeze(0) - init_params_b[..., 0].unsqueeze(1))
            radius_diff = torch.abs(tgt_params_b[..., 1].unsqueeze(0) - init_params_b[..., 1].unsqueeze(1))

            pos_line_list_b = []
            for gt_idx in range(n_gt):
                condition1 = theta_diff[:, gt_idx] < (self.cfg.threshold_theta / self.cfg.max_theta)
                condition2 = radius_diff[:, gt_idx] < (self.cfg.threshold_radius / self.cfg.max_radius)
                condition = torch.logical_and(condition1, condition2)
                pos_lines = torch.where(condition)[0]
                if len(pos_lines) == 0:
                    pos_lines = torch.argmin(radius_diff[:, gt_idx] ** 2 + theta_diff[:, gt_idx] ** 2, keepdim=True)
                pos_line_list_b.append(pos_lines)

            pos_line_list.append(torch.cat(pos_line_list_b))

        return pos_line_list

    # Combinations
    def score_weighted_regression_loss(self, pred_score, gt_score, score_mean=0.5):
        if pred_score.dim() > 1:
            pred_score = pred_score.reshape(-1)
        if gt_score.dim() > 1:
            gt_score = gt_score.reshape(-1)
        assert pred_score.shape == gt_score.shape, '{} vs. {}'.format(pred_score.shape, gt_score.shape)
        l1_loss = F.smooth_l1_loss(pred_score, gt_score, reduction='none')
        weight = torch.exp((gt_score - score_mean).clip(min=0, max=100))
        reg_loss = torch.mean(weight * l1_loss)

        return {'Reg': reg_loss}

    def score_rank_loss(self, pred_score, gt_score):
        rank_loss = 0
        assert pred_score.shape == gt_score.shape, '{} vs. {}'.format(pred_score.shape, gt_score.shape)
        B = pred_score.shape[0]
        N = pred_score.shape[1]
        pair_num = N * (N - 1) / 2
        for b in range(B):
            pre_diff = pred_score[b, :, None] - pred_score[b, None, :]
            gt_diff = gt_score[b, :, None] - gt_score[b, None, :]
            indicat = -1 * torch.sin(gt_diff) * (pre_diff - gt_diff)
            diff = torch.maximum(indicat, torch.zeros_like(indicat))
            rank_loss += torch.sum(diff) / pair_num

        return {'Ranking': rank_loss}


def get_loss_function(args):
    weight_dict = dict()
    weight_dict['loss_cls'] = args.loss_cls_coef
    weight_dict['loss_reg'] = args.loss_reg_coef
    weight_dict['loss_kld'] = args.loss_kld_coef

    losses = ['cls', 'reg']
    criterion = Loss_Function(args, weight_dict=weight_dict)

    return criterion
