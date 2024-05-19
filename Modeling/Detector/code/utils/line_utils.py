import cv2
import math
import torch
import numpy as np
from utils.util import to_tensor, to_tensor2, to_np
from utils.misc import coord_norm, coord_unnorm


# W H
def find_endpoints(data, size):
    x1, y1, x2, y2 = data[0], data[1], data[2], data[3]

    pts = []
    if x1 - x2 != 0:
        a = (y1 - y2) / (x1 - x2)
        b = -1
        c = -1 * a * x1 + y1
        # x = 0
        cx = 0
        cy = a * 0 + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)
        # x = size[0]
        cx = size[0]
        cy = a * size[0] + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)
        if len(pts) == 4:
            return np.float32(pts)
        # y = 0
        if y1 != y2:
            cx = (0 - c) / a
            cy = 0
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)
            if len(pts) == 4:
                return np.float32(pts)
            # y = size[1]
            cx = (size[1] - c) / a
            cy = size[1]
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)

    else:
        if x1 >= 0 and x1 <= size[0]:
            pts.append(x1)
            pts.append(0)
            pts.append(x1)
            pts.append(size[1])

    return np.float32(pts)

# Line equation -> Line points
def find_endpoints_from_line_eq(line_eq, size):
    a, b, c = line_eq

    pts = []
    if a == 1 and b == 0:
        x1 = c
        if x1 >= 0 and x1 <= size[0]:
            pts.append(x1)
            pts.append(0)
            pts.append(x1)
            pts.append(size[1])

    else:
        # x = 0
        cx = 0
        cy = a * 0 + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)
        # x = size[0]
        cx = size[0]
        cy = a * size[0] + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)

        if a != 0:
            # y = 0
            cx = (0 - c) / a
            cy = 0
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)

            # y = size[1]
            cx = (size[1] - c) / a
            cy = size[1]
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)

    pts = np.float32(pts)
    if pts.shape[0] != 0:
        pts = np.unique(pts.reshape(-1, 2), axis=0).reshape(-1)
    return pts

# Line points -> Line equation
def line_equation(data):
    # data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
    line_eq = np.zeros((data.shape[0], 3))   #.cuda()
    line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
    line_eq[:, 1] = -1
    line_eq[:, 2] = (-1 * line_eq[:, 0] * data[:, 0]) + data[:, 1]
    check = ((data[:, 0] - data[:, 2]) == 0)
    # check = (np.abs(data[:, 0] - data[:, 2]) < 0.5)
    line_eq = torch.FloatTensor(line_eq)
    return line_eq, check


# Line equation -> angle, dist
def transform_theta_to_angle(line_eq):
    line_angle = line_eq[:, 0].clone()
    line_angle = torch.atan(line_angle) * 180 / math.pi
    return line_angle


# line-point distance
def calculate_distance_from_center(line_eq, check, line_pts, center_pts):

    num = line_eq.shape[0]
    a = line_eq[:, 0].view(num, 1, 1)
    b = line_eq[:, 1].view(num, 1, 1)
    c = line_eq[:, 2].view(num, 1, 1)

    dist = (center_pts[0] * a + center_pts[1] * b + c) / torch.sqrt(a * a + b * b)

    if True in check:
        try:
            dist[check == True] = (center_pts[0] - line_pts[check == True, 0]).view(-1, 1, 1)
        except:
            dist[check == True] = torch.from_numpy((center_pts[0] - line_pts[check == True, 0]).reshape(-1, 1, 1)).float()

    return dist


# radius, theta -> Line points
def convert_to_line(height, width, radius, theta):
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    a = np.tan(theta / 180 * math.pi)

    if theta != -90:
        b = -1
        c = radius * np.sqrt(a ** 2 + b ** 2) - (a * center[0] + b * center[1])
    else:
        a = 1
        b = 0
        c = center[0] + radius

    line_pts = find_endpoints_from_line_eq(line_eq=[a, b, c], size=[width - 1, height - 1])
    return line_pts


def get_line_pts_from_normed_params(cfg, params, return_norm=True, eps=1e-4):
    # line params: [N, 2] -> line pts: [N, 4]
    reshape = False
    if len(params.shape) == 3:
        b, N, _ = params.shape
        params = params.reshape(b*N, -1)
        reshape = True

    hough_lines = params.clip(min=eps, max=1 - eps)
    spatial_lines = torch.zeros((0, 4))
    for idx, (theta, radius) in enumerate(to_np(hough_lines)):
        line = convert_to_line(cfg.height, cfg.width,
                               radius * (2 * cfg.max_radius) - cfg.max_radius,
                               theta * (2 * cfg.max_theta) - cfg.max_theta)
        if len(line) == 4:
            line = torch.from_numpy(line).view(1, 4)
        else:
            line = torch.tensor([0, 0, 0, 0]).view(1, 4)
        if return_norm:
            spatial_lines = torch.cat([spatial_lines, coord_norm(cfg, line)])
        else:
            spatial_lines = torch.cat([spatial_lines, line])

    if reshape:
        spatial_lines = spatial_lines.reshape(b, N, -1)
    return spatial_lines.cuda()


def get_line_pts_from_params(cfg, params, return_norm=True, eps=1e-4):
    # line params: [N, 2] -> line pts: [N, 4]
    reshape = False
    if len(params.shape) == 3:
        b, N, _ = params.shape
        params = params.reshape(b*N, -1)
        reshape = True

    hough_lines = params.clone()
    spatial_lines = torch.zeros((0, 4))
    for idx, (theta, radius) in enumerate(to_np(hough_lines)):
        line = convert_to_line(cfg.height, cfg.width, radius, theta)
        if len(line) == 4:
            line = torch.from_numpy(line).view(1, 4)
        else:
            line = torch.tensor([0, 0, 0, 0]).view(1, 4)
        if return_norm:
            spatial_lines = torch.cat([spatial_lines, coord_norm(cfg, line)])
        else:
            spatial_lines = torch.cat([spatial_lines, line])

    if reshape:
        spatial_lines = spatial_lines.reshape(b, N, -1)
    return spatial_lines.cuda()


def get_line_params(cfg, line_pts):
    # line_pts: [b, N, 4] -> line params: [b, N, 2]
    b, n, _ = line_pts.shape
    line_pts = line_pts.reshape(-1, 4)
    line_pts = to_np(line_pts)
    center = np.array([(cfg.width - 1) / 2, (cfg.height - 1) / 2])
    line_eq, check = line_equation(line_pts)
    theta = transform_theta_to_angle(line_eq)
    radius = calculate_distance_from_center(line_eq, check, line_pts, center)[:, 0, 0]
    angle_dist = torch.stack([theta, radius], dim=1).cuda()
    angle_dist = angle_dist.reshape(b, -1, 2)
    return angle_dist


def NMS(cfg, proposals_b, prob_b, threshold, maxk=5):
    select_idx = []

    prob = to_np(prob_b)
    proposals = to_np(proposals_b)
    center = np.array([(cfg.width - 1) / 2, (cfg.height - 1) / 2])

    sorted_idx = np.argsort(prob)[::-1]

    line_eq, check = line_equation(proposals)
    angle = transform_theta_to_angle(line_eq)
    dist = calculate_distance_from_center(line_eq, check, proposals, center)[:, 0, 0]

    angle_dist = torch.stack([angle, dist], dim=1)
    angle_dist[..., 0] = angle_dist[..., 0] / (cfg.max_theta / 2)     # More sensitive on angle displacement
    angle_dist[..., 1] = angle_dist[..., 1] / cfg.max_radius
    visit_mask = torch.ones(len(proposals), dtype=torch.int64)
    count = 0

    assert len(sorted_idx) == len(angle_dist)

    for idx in range(len(angle_dist)):
        now = sorted_idx[idx]
        if count == maxk:
            break
        if visit_mask[now] == 0:
            continue
        if prob[now] < cfg.prob_threshold:
            if len(select_idx) == 0:
                select_idx.append(now)
            return torch.LongTensor(select_idx)
        dist = angle_dist - angle_dist[now]
        cost = (dist ** 2).mean(-1)
        cost[now] = 10000
        delete_idx = torch.where(cost < threshold)[0]
        visit_mask[delete_idx] = 0
        visit_mask[now] = 0

        count += 1
        select_idx.append(now)

    return torch.LongTensor(select_idx)
