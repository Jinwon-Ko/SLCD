import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy.optimize import linear_sum_assignment

from models.tools.matcher import build_matcher
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
    def __init__(self, cfg, matcher, weight_dict):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = 1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = cfg.focal_alpha

        self.loss_focal = SoftmaxFocalLoss(gamma=2)

    # Losses
    def get_loss_for_line_detection(self, outputs, targets):
        loss_cls, loss_reg = self.loss_for_nms_version(outputs, targets)
        return {'Cls': loss_cls * self.weight_dict['loss_cls'],
                'Reg': loss_reg * self.weight_dict['loss_reg']}

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


def get_loss_function(args):
    matcher = build_matcher(args)

    weight_dict = dict()
    weight_dict['loss_cls'] = args.loss_cls_coef
    weight_dict['loss_reg'] = args.loss_reg_coef

    criterion = Loss_Function(args, matcher=matcher, weight_dict=weight_dict)

    return criterion
