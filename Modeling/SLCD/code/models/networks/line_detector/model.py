import torch
import torch.nn as nn

from models.tools.network_modules import Feature_Extractor, Line_Pooling_norm, Cross_Attn, Prediction_Heads
from models.tools.network_tools import Make_features, Post_Process

from utils.misc import coord_unnorm, coord_norm
from utils.line_utils import get_line_params, get_line_pts_from_params, get_line_pts_from_normed_params


def init_query_lines(spatial_shapes):
    h, w = spatial_shapes

    grid_y, grid_x = torch.meshgrid(torch.linspace(0, h - 1, h, dtype=torch.float32, device=h.device),
                                    torch.linspace(0, w - 1, w, dtype=torch.float32, device=h.device))
    grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

    scale = spatial_shapes.view(1, 1, 2)
    grid = (grid + 0.5) / scale

    proposal = grid.view(-1, 2)
    # proposal = proposal[(0.02 < proposal[:, 1]) * (proposal[:, 1] < 0.96), :]
    proposal = proposal[(0.05 < proposal[:, 1]) * (proposal[:, 1] < 0.95), :]

    return proposal


class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg

        # Networks
        self.feature_extractor = Feature_Extractor(cfg)
        self.scale_factor = self.feature_extractor.scale_factor[0]

        self.line_pool_layer = Line_Pooling_norm(cfg, self.cfg.hidden_dim)
        self.prediction_heads = Prediction_Heads(cfg, self.cfg.hidden_dim)

        self.line_representation = Make_features(cfg)

        spatial_shapes = torch.as_tensor([self.cfg.num_anchors ** 0.5, self.cfg.num_anchors ** 0.5], dtype=torch.long).cuda()
        self.init_params = init_query_lines(spatial_shapes)      # [N, 2]
        self.init_pts = get_line_pts_from_normed_params(cfg, self.init_params, return_norm=False)
        self.init_line_mask, self.init_n_l = self.line_representation.get_line_masks(self.init_params, self.scale_factor, margin=0.5)  # [N, h, w]
        self.gaussian_dist_map = self.line_representation.generate_gaussian_dist_map(self.init_params, self.scale_factor)   # [N, h, w]

        self.topk = cfg.topk
        self.post_processor = Post_Process(cfg)

    def forward(self, img, targets=None):
        outputs = dict()
        self.inner_dicts = dict()
        self.forward_for_feature_extraction(img)
        outputs.update(self.forward_for_semantic_line_detection())
        return outputs

    def forward_for_feature_extraction(self, img):
        img_feat, img_pos = self.feature_extractor(img)      # [b, c, h, w]

        self.img_feat = img_feat        # [b, c, h, w]
        self.img_pos = img_pos          # [b, c, h, w]
        self.batch_size = img.shape[0]

    def forward_for_semantic_line_detection(self):
        anchor_pts = self.init_pts.repeat(self.batch_size, 1, 1)            # [b, n, 4]
        anchor_params = self.init_params.repeat(self.batch_size, 1, 1)      # [b, n, 2]

        result = dict()
        result['pts_init'] = anchor_pts
        result['params_init'] = anchor_params

        anchor_feat = self.line_pool_layer(self.img_feat, coord_norm(self.cfg, anchor_pts))    # [b, n, c]
        # anchor_feat = self.get_line_feats(self.img_feat)  # [b, n, c]
        # anchor_feat = self.get_region_features(self.img_feat)  # [b, n, c]
        anchor_logit, anchor_reg = self.prediction_heads(anchor_feat)

        # Update Cls / Reg
        anchor_params = anchor_params + anchor_reg
        anchor_pts = get_line_pts_from_normed_params(self.cfg, anchor_params, return_norm=False)

        result[f'cls'] = anchor_logit
        result[f'prob'] = torch.softmax(anchor_logit, dim=-1)
        result[f'pts'] = anchor_pts
        result[f'params'] = anchor_params

        return result

    def get_line_feats(self, feats):
        line_mask = self.init_line_mask.repeat(self.batch_size, 1, 1, 1)         # [b, N, h, w]
        n_l = self.init_n_l.repeat(self.batch_size, 1)                           # [b, N]
        line_feats = torch.sum(feats.unsqueeze(1) * line_mask.unsqueeze(2), dim=[-2, -1]) / n_l.unsqueeze(-1)
        return line_feats       # [b, N, c]

    def get_region_features(self, feats):
        b, c, h, w = feats.shape
        gaussian_dist_map = self.gaussian_dist_map.repeat(b, 1, 1, 1)           # [b, N, h, w]
        region_feat = torch.sum(feats.unsqueeze(1) * gaussian_dist_map.unsqueeze(2), dim=(3, 4))
        return region_feat      # [b, N, c]

    def prepare_for_combination(self, outputs):
        line_pts = outputs['pts'].clone().detach()
        line_params = outputs['params'].clone().detach()
        probs = outputs['prob'][..., 1].clone().detach()
        selected_idx = self.post_processor.detect_topk(line_pts, probs)

        topk_idxes = torch.cat([torch.arange(len(probs)).reshape(-1, 1) for _ in range(self.topk)], dim=1)
        topk_probs = probs[topk_idxes, selected_idx]
        topk_line_pts = line_pts[topk_idxes, selected_idx, :]
        topk_line_params = line_params[topk_idxes, selected_idx, :]

        comb_outputs = dict()
        comb_outputs['topk_prob'] = topk_probs.clone().detach()
        comb_outputs['topk_line_pts'] = topk_line_pts.clone().detach()
        comb_outputs['topk_line_params'] = topk_line_params.clone().detach()
        return comb_outputs
