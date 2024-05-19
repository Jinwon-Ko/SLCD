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
    # img = cv2.imread(r'D:/NAVER2021\Sline_sample\data\%s'%(dataname[10]))

    # Start coordinate, represents the top left corner of image
    start_point = (int(pts[0]), int(pts[1]))

    # End coordinate, represents the bottom right corner of image
    end_point = (int(pts[2]), int(pts[3]))

    # Using cv2.line() method
    image = cv2.line(img, start_point, end_point, color, thickness)
    return image


def viz_line(img, pts, color=(0, 255, 0), thickness = 7):
    # img = cv2.imread(r'D:/NAVER2021\Sline_sample\data\%s'%(dataname[10]))

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

    def show_image(self):
        plt.figure()
        plt.imshow(self.show['img'][:, :, [2, 1, 0]])

    def save_fig(self, path, name):
        img_path, img_name = '/'.join(name.split('/')[:-1]), name.split('/')[-1]
        save_path = os.path.join(path, img_path)
        os.makedirs(save_path, exist_ok=True)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, img_name), bbox_inches='tight', pad_inches=0)
        plt.close()

    def draw_lines_plt(self, endpts, linewidth=2, color='yellow', linestyle='-', zorder=1):
        if len(endpts) == 0:
            return
        pt_1 = (endpts[0], endpts[1])
        pt_2 = (endpts[2], endpts[3])
        plt.plot([pt_1[0], pt_2[0]], [pt_1[1], pt_2[1]],
                 linestyle=linestyle,
                 linewidth=linewidth,
                 color=color,
                 zorder=zorder)

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

    def viz_train(self, cfg, img, outputs, gt_line, targets, criterion, b_idx, num):
        ######################################################################################
        # Draw input image & GT
        self.update_image(img=img)
        self.draw_line(to_np(gt_line), name='gt', color=(0, 255, 0), thickness=4)

        topk_lines = outputs[f'topk_line_pts'][b_idx]
        gt_best_comb_idx = outputs['gt_score'][b_idx, :].argmax(-1)
        gt_best_comb_list = outputs['comb_list'][b_idx, gt_best_comb_idx]
        gt_comb_lines = topk_lines[gt_best_comb_list]

        pred_best_comb_idx = outputs['pred_score'][b_idx, :, 0].argmax(-1)
        pred_best_comb_list = outputs['comb_list'][b_idx, pred_best_comb_idx]
        pred_comb_lines = topk_lines[pred_best_comb_list]
        self.draw_line(to_np(topk_lines), name='topk', color=(0, 180, 255), thickness=4)
        self.draw_line(to_np(gt_comb_lines), name='gt_comb', color=(0, 180, 255), thickness=4)
        self.draw_line(to_np(pred_comb_lines), name='pred_comb', color=(0, 255, 255), thickness=4)

        show_list = ['gt', 'topk', 'gt_comb', 'pred_comb']
        dir_name = os.path.join(cfg.viz_dir, f'train/comb_detection')
        self.saveimg(dir_name=dir_name, file_name=f'iter_{num:04d}.jpg', show_list=show_list)

    def viz_test(self, cfg, img, targets, results, img_name, epoch):
        ######################################################################################
        # Draw input image & GT
        self.update_image(img=img)
        self.draw_line(to_np(targets), name='gt', color=(0, 255, 0), thickness=4)

        topk_lines = results['topk_lines']
        gt_comb_lines = results['gt_comb']
        pred_comb_lines = results['pred_comb']
        self.draw_line(to_np(topk_lines), name='topk', color=(0, 180, 255), thickness=4)
        self.draw_line(to_np(gt_comb_lines), name='gt_comb', color=(0, 180, 255), thickness=4)
        self.draw_line(to_np(pred_comb_lines), name='pred_comb', color=(0, 255, 255), thickness=4)

        dir_name = os.path.join(cfg.viz_dir, f'test/results/epoch_{epoch:02d}')
        self.saveimg(dir_name=dir_name, file_name=img_name, show_list=['gt', 'topk', 'gt_comb', 'pred_comb'])

    def viz_analysis(self, cfg, img, targets, results, img_name):
        ######################################################################################
        # Draw input image & GT
        self.update_image(img=img)
        self.draw_line(to_np(targets), name='gt', color=(0, 255, 0), thickness=4)

        topk_lines = results['topk_lines']
        gt_comb_lines = results['gt_comb']
        pred_comb_lines = results['pred_comb']
        self.draw_line(to_np(topk_lines), name='topk', color=(0, 180, 255), thickness=4)
        self.draw_line(to_np(gt_comb_lines), name='gt_comb', color=(0, 180, 255), thickness=4)
        self.draw_line(to_np(pred_comb_lines), name='pred_comb', color=(0, 255, 255), thickness=4)
        ######################################################################################
        # Save visualize results
        dir_name = os.path.join(cfg.viz_dir, f'analysis/results')
        self.saveimg(dir_name=dir_name, file_name=img_name, show_list=['gt', 'topk', 'gt_comb', 'pred_comb'])

    def viz_forward_detector(self, cfg, img, gt, topk_lines, img_name, dataset):
        self.update_image(img=img)
        self.draw_line(to_np(gt), name='gt', color=(0, 255, 0), thickness=4)
        self.draw_line(to_np(topk_lines), name='topk', color=(0, 180, 255), thickness=4)  # Draw all proposal
        dir_name = os.path.join(cfg.viz_dir, f'forward_detector/topk_results/{dataset}')
        self.saveimg(dir_name=dir_name, file_name=img_name, show_list=['gt', 'topk'])

    def viz_cross_attn_pattern(self, cfg, cross_attn, img_name):
        show_list = ['gt', 'pred']
        ######################################################################################
        # Draw cross_attn
        spatial = cross_attn.shape[-1]
        attn_weights = cross_attn[0].reshape(-1, int(spatial**0.5), int(spatial**0.5))
        assign_map = attn_weights.argmax(0) / 7
        # for pattern_idx, attn_weight in enumerate(attn_weights):
        #     attn_weight = cv2.resize(to_np(attn_weight), dsize=(cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)
        #     self.draw_attn_map(mask=attn_weight, name=f'cross_attn_{pattern_idx}', alpha=0., colormap='jet')
        #     show_list.append(f'cross_attn_{pattern_idx}')

        assign_map = cv2.resize(to_np(assign_map), dsize=(cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
        self.draw_attn_map(mask=assign_map, name=f'cross_attn_map', alpha=0.5, colormap='rainbow')
        show_list.append(f'cross_attn_map')
        ######################################################################################
        # Save visualize results
        dir_name = os.path.join(cfg.viz_dir, f'analysis/cross_attn_pattern')
        self.saveimg(dir_name=dir_name, file_name=img_name, show_list=show_list)
