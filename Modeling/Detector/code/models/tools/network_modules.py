import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.networks.backbone import ResNet50
from models.networks.position_encoding import PositionEmbeddingSine


# Basic Layers
class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Network Blocks
class Feature_Extractor(nn.Module):
    def __init__(self, cfg):
        super(Feature_Extractor, self).__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim

        # Feature Extraction
        self.backbone = ResNet50(cfg)
        self.num_channels = self.backbone.num_channels
        self.scale_factor = self.backbone.scale_factor

        self.feat_squeeze1 = nn.Sequential(
            conv_bn_relu(self.num_channels[0], self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2))
        self.feat_squeeze2 = nn.Sequential(
            conv_bn_relu(self.num_channels[1], self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2))
        self.feat_squeeze3 = nn.Sequential(
            conv_bn_relu(self.num_channels[2], self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2))

        self.feature_combine = nn.Sequential(
            conv_bn_relu(self.hidden_dim * len(self.num_channels), self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2),
            conv_bn_relu(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, dilation=2, padding=2),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1))

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.cfg.hidden_dim // 2, normalize=True)

    def forward(self, img):
        f3, f4, f5 = self.backbone(img)

        f3 = self.feat_squeeze1(f3)
        f4 = self.feat_squeeze2(f4)
        f5 = self.feat_squeeze3(f5)

        f4 = F.interpolate(f4, scale_factor=2.0, mode='bilinear')
        f5 = F.interpolate(f5, scale_factor=4.0, mode='bilinear')

        x_concat = torch.cat([f3, f4, f5], dim=1)
        img_feats = self.feature_combine(x_concat)
        img_pos = self.pos_embed(img_feats)
        return img_feats, img_pos


class Line_Pooling_norm(nn.Module):
    def __init__(self, cfg, in_dim, step=64):
        super(Line_Pooling_norm, self).__init__()
        self.cfg = cfg

        self.step = step
        self.f_size = int(math.sqrt(self.step))
        self.size = torch.FloatTensor([self.cfg.width, self.cfg.height, self.cfg.width, self.cfg.height]).cuda()
        self.dim = step * in_dim

        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, in_dim, bias=False))
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, feat_map, line_pts):
        N = line_pts.shape[1]               # [b, N, 4]

        line_pts = (line_pts - 0.5) * 2     # normalized to [-1, 1]

        grid_X = line_pts[:, :, [0, 2]]     # Width    [b, N, 2]
        grid_Y = line_pts[:, :, [1, 3]]     # Height   [b, N, 2]

        line_X = F.interpolate(grid_X, self.step, mode='linear', align_corners=True)  # [b, N, step]
        line_Y = F.interpolate(grid_Y, self.step, mode='linear', align_corners=True)  # [b, N, step]

        line_X = line_X.view(-1, N, self.f_size, self.f_size)   # [b, N, f_size, f_size]
        line_Y = line_Y.view(-1, N, self.f_size, self.f_size)   # [b, N, f_size, f_size]
        line_grid = torch.stack((line_X, line_Y), dim=-1)       # [b, N, f_size, f_size, 2]

        cat = []
        for b_idx in range(len(feat_map)):
            f_lp = F.grid_sample(feat_map[b_idx:b_idx+1].repeat(N, 1, 1, 1), line_grid[b_idx])
            cat.append(f_lp)

        f_lp_all = torch.stack(cat, dim=0)
        f_lp_all = f_lp_all.view(len(feat_map), N, -1)
        f_lp_agg = self.fc(f_lp_all)            # [b, N, c]
        f_lp_agg = self.layer_norm(f_lp_agg)
        return f_lp_agg         # [b, N, c]


class Cross_Attn(nn.Module):
    def __init__(self, cfg):
        super(Cross_Attn, self).__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.dim_feedforward = cfg.dim_feedforward
        self.dropout = nn.Dropout(self.cfg.dropout)

        self.cross_q_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_k_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.cross_v_embed = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        self.ffn1 = nn.Sequential(nn.Linear(self.hidden_dim, self.dim_feedforward),
                                  nn.ReLU())
        self.ffn2 = nn.Sequential(nn.Linear(self.dim_feedforward, self.hidden_dim))
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self._reset_parameters()

        # self.register_buffer('cross_attn_weight', None)

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_feat, img_pos, pattern_feat):
        """
        Internal Discretization
        :param img_feat: Tensor of dim [b, h*w, c] with the image features
        :param img_pos: Tensor of dim [b, h*w, c] with the image positional embedding
        :param pattern_feat: Tensor of dim [b, m, c] with the pattern features
        :return:
        """
        # Cross-Attn
        q = self.cross_q_embed(pattern_feat)            # [b, m, c]
        k = self.cross_k_embed(img_feat + img_pos)      # [b, h*w, c]
        v = self.cross_v_embed(img_feat + img_pos)      # [b, h*w, c]
        # Softmax column direction
        cross_attn_weight = torch.softmax(q @ k.transpose(-2, -1), dim=-2)  # [b, m, h*w]

        # Register buffer
        self.cross_attn_weight = cross_attn_weight

        # cross_attn_weight = cross_attn_weight / torch.sum(cross_attn_weight, dim=-1, keepdim=True)  # normalize (divide into # of assigned pixels)
        x = cross_attn_weight @ v                       # [b, m, c]
        pattern_feat = self.norm1(pattern_feat + x)     # [b, m, c]

        # FFN
        x2 = self.ffn1(pattern_feat)
        x2 = self.ffn2(self.dropout(x2))
        pattern_feat = pattern_feat + self.dropout(x2)
        pattern_feat = self.norm2(pattern_feat)

        return pattern_feat


class Prediction_Heads(nn.Module):
    def __init__(self, cfg, feat_dim):
        super(Prediction_Heads, self).__init__()
        self.cfg = cfg
        self.query_dim = cfg.query_dim

        self.cls_head = nn.Sequential(nn.Linear(feat_dim, feat_dim),
                                      nn.ReLU(),
                                      nn.Linear(feat_dim, 2))
        self.reg_head = nn.Sequential(nn.Linear(feat_dim, feat_dim),
                                      nn.ReLU(),
                                      nn.Linear(feat_dim, self.query_dim))

        nn.init.normal_(self.reg_head[-1].weight.data, mean=0, std=0.005)
        nn.init.normal_(self.reg_head[-1].bias.data, mean=0, std=0.005)

    def forward(self, query_feats):
        query_cls = self.cls_head(query_feats)
        query_reg = self.reg_head(query_feats)
        return query_cls, query_reg


class Self_Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Self_Attention, self).__init__()
        self.self_q_embed = nn.Linear(in_dim, out_dim)
        self.self_k_embed = nn.Linear(in_dim, out_dim)
        self.self_v_embed = nn.Linear(in_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat):
        q = self.self_q_embed(feat)                 # [b, h*w, c]
        k = self.self_k_embed(feat)                 # [b, h*w, c]
        v = self.self_v_embed(feat)                 # [b, h*w, c]

        # Softmax row direction
        self_attn_weight = torch.softmax(q @ k.transpose(-2, -1), dim=-1)  # [b, h*w, h*w]
        x = self_attn_weight @ v                    # [b, h*w, c]
        out = self.norm(feat + x)                   # [b, h*w, c]
        return out

class Cross_Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Cross_Attention, self).__init__()
        self.cross_q_embed = nn.Linear(in_dim, out_dim)
        self.cross_k_embed = nn.Linear(in_dim, out_dim)
        self.cross_v_embed = nn.Linear(in_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, img_feat, comb_feat):
        q = self.cross_q_embed(comb_feat)           # [b, h*w, c]
        k = self.cross_k_embed(img_feat)            # [b, h*w, c]
        v = self.cross_v_embed(img_feat)            # [b, h*w, c]

        # Softmax row direction
        cross_attn_weight = torch.softmax(q @ k.transpose(-2, -1), dim=-1)  # [b, h*w, h*w]
        x = cross_attn_weight @ v                   # [b, h*w, c]
        out = self.norm(comb_feat + x)              # [b, h*w, c]
        return out
