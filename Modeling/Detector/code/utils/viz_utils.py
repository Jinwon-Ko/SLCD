import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.util import to_np
from utils.line_utils import NMS, get_line_pts_from_normed_params

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


############################################
def viz_GT(img, pts, color=(0, 255, 0), thickness = 7):
    # Start coordinate, represents the top left corner of image
    start_point = (int(pts[0]), int(pts[1]))

    # End coordinate, represents the bottom right corner of image
    end_point = (int(pts[2]), int(pts[3]))

    # Using cv2.line() method
    image = cv2.line(img, start_point, end_point, color, thickness)
    return image


def viz_line(img, pts, color=(0, 255, 0), thickness = 7):
    # Start coordinate, represents the top left corner of image
    start_point = (int(pts[0]), int(pts[1]))

    # End coordinate, represents the bottom right corner of image
    end_point = (int(pts[2]), int(pts[3]))

    # Using cv2.line() method
    image = cv2.line(img, start_point, end_point, color, thickness)
    return image


def draw_text(img, pred, color=(255, 0, 0)):
    cv2.rectangle(img, (1, 1), (250, 120), color, 1)
    cv2.putText(img, 'pred : ' + str(pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

############################################
class Visualizer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def draw_line(self, line_pts, name, ref_name='img', color=(0, 255, 0), thickness=2):
        """ pts : [N, 4] """
        line_pts = line_pts.reshape(-1, 4)
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for idx in range(len(line_pts)):
            pts = line_pts[idx]
            img = viz_line(img.copy(), pts, color=color, thickness=thickness)
        self.show[name] = img

    def draw_attn_map(self, mask, name, ref_name='img', alpha=0.5, colormap='rainbow'):
        img = np.ascontiguousarray(np.copy(self.show[ref_name])) / 255
        if colormap == 'rainbow':
            heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_RAINBOW)
        elif colormap == 'jet':
            heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap * (1 + alpha) + img * (1 - alpha)
        cam = cam / np.max(cam)
        self.show[name] = np.uint8(255 * cam)

    def saveimg(self, dir_name, file_name, show_list):
        # boundary line
        if self.show[show_list[0]].shape[0] != self.line.shape[0]:
            self.line = np.zeros((self.show[show_list[0]].shape[0], 3, 3), dtype=np.uint8)
            self.line[:, :, :] = 255
        disp = self.line

        for i in range(len(show_list)):
            if show_list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[show_list[i]], self.line), axis=1)

        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def saveimg_one(self, dir_name, file_name, show_list):
        disp = self.show[show_list[0]]
        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def viz_train(self, cfg, img, outputs, gt_line, targets, criterion, b_idx, num):
        ######################################################################################
        # Draw input image & GT
        self.update_image(img=img)
        self.draw_line(to_np(gt_line), name='gt', color=(0, 255, 0), thickness=4)

        lines = get_line_pts_from_normed_params(cfg, outputs[f'params'][b_idx], return_norm=False)
        pos_lines = criterion.get_pos_lines(outputs, targets)
        matched_idx = pos_lines[b_idx]
        self.draw_line(to_np(lines[matched_idx]), name=f'pos', color=(255, 0, 0), thickness=4)

        probas = outputs[f'prob'][b_idx, :, 1]
        keep = NMS(cfg, lines, probas, threshold=cfg.test_nms_threshold)
        self.draw_line(to_np(lines[keep]), name=f'pred', color=(0, 255, 255), thickness=4)  # Draw init lines

        topk_lines = get_line_pts_from_normed_params(cfg, outputs[f'topk_line_params'][b_idx], return_norm=False)
        self.draw_line(to_np(topk_lines), name=f'topk', color=(0, 180, 255), thickness=4)  # Draw init lines

        show_list = ['gt', 'pos', 'pred', 'topk']
        dir_name = os.path.join(cfg.viz_dir, f'train/line_detection')
        self.saveimg(dir_name=dir_name, file_name=f'iter_{num:04d}.jpg', show_list=show_list)

    def viz_test(self, cfg, img, targets, results, img_name, epoch):
        ######################################################################################
        # Draw input image & GT
        self.update_image(img=img)
        self.draw_line(to_np(targets), name='gt', color=(0, 255, 0), thickness=4)

        lines = results['lines']
        keep = results['keep']
        topk_lines = results['topk_line_pts']
        self.draw_line(to_np(lines), name='proposals', color=(0, 180, 255), thickness=2)
        self.draw_line(to_np(lines[keep]), name='pred', color=(0, 255, 255), thickness=4)
        self.draw_line(to_np(topk_lines), name='topk', color=(0, 180, 255), thickness=2)  # Draw all proposal
        ######################################################################################
        # Save visualize results
        dir_name = os.path.join(cfg.viz_dir, f'test/results/epoch_{epoch:02d}')
        self.saveimg(dir_name=dir_name, file_name=img_name, show_list=['gt', 'pred', 'topk'])

    def viz_analysis(self, cfg, img, targets, results, img_name):
        ######################################################################################
        # Draw input image & GT
        self.update_image(img=img)
        self.draw_line(to_np(targets), name='gt', color=(0, 255, 0), thickness=4)

        lines = results['lines']
        keep = results['keep']
        topk_lines = results['topk_line_pts']
        self.draw_line(to_np(lines), name='proposals', color=(0, 180, 255), thickness=2)
        self.draw_line(to_np(lines[keep]), name='pred', color=(0, 255, 255), thickness=4)
        self.draw_line(to_np(topk_lines), name='topk', color=(0, 180, 255), thickness=2)  # Draw all proposal
        # ######################################################################################
        # Save visualize results
        dir_name = os.path.join(cfg.viz_dir, f'analysis/results')
        self.saveimg(dir_name=dir_name, file_name=img_name, show_list=['gt', 'pred', 'topk'])
