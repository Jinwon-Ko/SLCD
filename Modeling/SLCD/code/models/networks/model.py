import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from models.tools.network_modules import Feature_Extractor, Line_Pooling_norm, Cross_Attn, Prediction_Heads, \
    Combination_Conv, Combination_Heads, Comb_Position_Embed
from models.tools.network_tools import Make_features, Post_Process

from utils.misc import coord_unnorm
from utils.line_utils import get_line_params, get_line_pts_from_normed_params


def init_query_lines(spatial_shapes):
    h, w = spatial_shapes

    grid_y, grid_x = torch.meshgrid(torch.linspace(0, h - 1, h, dtype=torch.float32, device=h.device),
                                    torch.linspace(0, w - 1, w, dtype=torch.float32, device=h.device))
    grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

    scale = spatial_shapes.view(1, 1, 2)
    grid = (grid + 0.5) / scale

    proposal = grid.view(-1, 2)
    proposal = proposal[(0.05 < proposal[:, 1]) * (proposal[:, 1] < 0.95), :]

    return proposal


class SLCD(nn.Module):
    def __init__(self, cfg):
        super(SLCD, self).__init__()
        self.cfg = cfg
        self.topk = cfg.topk
        self.post_processor = Post_Process(cfg)

        # Networks
        self.feature_extractor = Feature_Extractor(cfg)
        self.scale_factor = self.feature_extractor.scale_factor[0]

        self.cross_attn = nn.ModuleList([Cross_Attn(cfg) for _ in range(self.cfg.n_dec_layers)])
        self.pattern_embed = nn.Embedding(self.cfg.num_region_queries, self.cfg.hidden_dim)   # [m, c]
        nn.init.normal_(self.pattern_embed.weight.data)

        if self.cfg.use_semantic_feat:
            self.combi_pe = Comb_Position_Embed(self.topk, self.cfg.hidden_dim * 2)
            self.combi_conv = Combination_Conv(self.cfg.hidden_dim * 6, self.cfg.hidden_dim)
        else:
            self.combi_pe = Comb_Position_Embed(self.topk, self.cfg.hidden_dim)
            self.combi_conv = Combination_Conv(self.cfg.hidden_dim * 3, self.cfg.hidden_dim)
        self.combination_score_reg = Combination_Heads(cfg, self.cfg.hidden_dim*225, self.cfg.hidden_dim*15, 1)

        self.line_representation = Make_features(cfg)
        spatial_shapes = torch.as_tensor([self.cfg.num_anchors ** 0.5, self.cfg.num_anchors ** 0.5], dtype=torch.long).cuda()
        self.init_params = init_query_lines(spatial_shapes)      # n, 2
        self.init_pts = get_line_pts_from_normed_params(cfg, self.init_params, return_norm=False)
        self.init_line_mask, self.init_n_l = self.line_representation.get_line_masks(self.init_params, self.scale_factor)   # [N, h, w]
        self.gaussian_dist_map = self.line_representation.generate_gaussian_dist_map(self.init_params, self.scale_factor)   # [N, h, w]

    def freeze_params_(self):
        for name, p in self.named_parameters():
            p.requires_grad = False

    def forward(self, img, detector_outputs, targets=None):
        outputs = dict()
        outputs.update(detector_outputs)
        comb_mask = detector_outputs['comb_mask']
        positional_map = detector_outputs['positional_map'].float()

        self.forward_for_feature_extraction(img)
        outputs.update(self.forward_for_semantic_partitioning())
        self.forward_for_generate_semantic_feature()
        outputs.update(self.forward_for_combination_score_regresssion(comb_mask, positional_map))
        outputs.update(self.prepare_for_KLDloss(targets))
        return outputs

    def prepare_for_KLDloss(self, targets=None):
        if targets is None:
            return {}
        else:
            b, c, h, w = self.img_feat.shape
            b, T, _ = targets['params'].shape
            sf = self.cfg.height // h

            tgt_line = torch.cat([v[v[:, 0] != -1000] for v in targets['params']])
            sizes = [len(torch.nonzero(v[:, 0] != -1000)) for v in targets['params']]
            zero_sizes = [len(torch.nonzero(v[:, 0] == -1000)) for v in targets['params']]

            region_mask = self.line_representation.get_region_masks(tgt_line, sf)  # [b*N, h, w]

            zero_mask = torch.zeros(b * T - len(region_mask), h, w, device=region_mask.device)
            mask_r = region_mask.split(sizes, 0)
            zero_r = zero_mask.split(zero_sizes, 0)
            positional_map = torch.stack([torch.cat([mask_r[i], zero_r[i]]) for i in range(b)])
            return {'positional_map': positional_map}

    def forward_for_feature_extraction(self, img):
        img_feat, img_pos = self.feature_extractor(img)      # [b, c, h, w]

        self.img_feat = img_feat        # [b, c, h, w]
        self.img_pos = img_pos          # [b, c, h, w]
        self.batch_size = img.shape[0]

    def forward_for_semantic_partitioning(self):
        pattern_feat = self.pattern_embed.weight.repeat(self.batch_size, 1, 1)  # [b, m, c]
        img_feat = self.img_feat.flatten(2).permute(0, 2, 1)                    # [b, h*w, c]
        img_pos = self.img_pos.flatten(2).permute(0, 2, 1)                      # [b, h*w, c]

        # Cross attention
        cross_attn_weight = []
        for layer_idx in range(self.cfg.n_dec_layers):
            pattern_feat = self.cross_attn[layer_idx](img_feat=img_feat, img_pos=img_pos, pattern_feat=pattern_feat)
            cross_attn_weight.append(self.cross_attn[layer_idx].cross_attn_weight)

        self.membership_map = cross_attn_weight[-1].transpose(-2, -1)
        self.pattern_feat = pattern_feat
        return {f'attn_map': torch.stack(cross_attn_weight)}

    def forward_for_generate_semantic_feature(self):
        b, c, h, w = self.img_feat.shape
        img_feat = self.img_feat.flatten(2).permute(0, 2, 1)            # [b, h*w, c]

        membership_map = self.membership_map                            # [b, h*w, m]
        pattern_feat = self.pattern_feat                                # [b, m, c]
        region_feat = membership_map @ pattern_feat                     # [b, h*w, c]
        semantic_feat = torch.cat([img_feat, region_feat], dim=-1)      # [b, h*w, 2*c]

        self.region_feat = region_feat
        self.semantic_feat = semantic_feat.view(b, h, w, -1).permute(0, 3, 1, 2)

    def forward_for_combination_score_regresssion(self, comb_mask, positional_map):
        if self.cfg.use_semantic_feat:
            feat_map = self.semantic_feat    # [self.img_feat, self.semantic_feat]
        else:
            feat_map = self.img_feat

        b, c, h, w = feat_map.shape
        b, s, h, w = comb_mask.shape

        region_mask = torch.logical_not(comb_mask)
        positional_map = positional_map.reshape(b*s, self.topk, h, w)               # [b*s, k, h, w]
        positional_emb = self.combi_pe(positional_map)                              # [b*s, c, h, w]

        comb_mask = comb_mask.view(b, s, 1, h, w)
        region_mask = region_mask.view(b, s, 1, h, w)
        line_feat = feat_map.view(b, 1, c, h, w) * comb_mask                        # [b, s, c, h, w]
        region_feat = feat_map.view(b, 1, c, h, w) * region_mask                    # [b, s, c, h, w]

        line_feat = line_feat.reshape(b*s, c, h, w)                                 # [b*s, c, h, w]
        region_feat = region_feat.reshape(b*s, c, h, w)                             # [b*s, c, h, w]
        comb_feats = torch.cat([line_feat, region_feat, positional_emb], dim=1)     # [b*s, 3*c, h, w]

        comb_feats = F.interpolate(comb_feats, scale_factor=1 / 4, mode='bilinear') # [b*s, 3*c, h/4, w/4]
        comp_feats = self.combi_conv(comb_feats)                                    # [b*s, c, h/4, w/4]
        comb_feats = comp_feats.reshape(b, s, -1)                                   # [b, s, c*h*w/16]

        comb_score = self.combination_score_reg(comb_feats)
        pred_best_comb_idx = torch.argmax(comb_score[..., 0], dim=-1)
        comp_feat = comp_feats.reshape(b, s, self.cfg.hidden_dim, h//4, w//4)[:, pred_best_comb_idx]
        return {'pred_score': comb_score, 'pred_best_comb_idx': pred_best_comb_idx, 'compositional_feat': comp_feat.squeeze()}
