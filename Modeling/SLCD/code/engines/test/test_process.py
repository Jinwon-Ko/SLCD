import os
import time
import torch
import pickle
import numpy as np

from utils.util import to_np, save_best_line_detection_model
from utils.misc import coord_norm, coord_unnorm
from utils.viz_utils import Visualizer

from utils.calculate_metrics import Evaluation_Semantic_Line


def test_line_detection_with_combination(cfg, epoch, model, test_loader, criterion, optimizer, best_HIoU):
    if not epoch >= cfg.start_eval_epoch:
        return best_HIoU
    model.eval()
    criterion.eval()

    eval_line = Evaluation_Semantic_Line(cfg)
    viz_tools = Visualizer(cfg)
    hiou, max_hiou = 0, 0
    match1 = {'p': {}, 'r': {}}
    EA_scores = {'pred': {}, 'gt': {}}
    for i, batch in enumerate(test_loader):
        print('[Epoch: %d][%d/%d]' % (epoch, i, len(test_loader)), end='\r')

        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        for k, v in batch['detector_outputs'].items():
            batch['detector_outputs'][k] = v.cuda()

        targets = {'pts': gt_pts, 'params': gt_params}

        with torch.no_grad():
            outputs = model(img, batch['detector_outputs'], targets=targets)

        topk_lines = outputs[f'topk_line_pts'][0]
        gt_best_comb_idx = outputs['gt_score'][0, :].argmax(-1)
        gt_best_comb_list = outputs['comb_list'][0, gt_best_comb_idx]
        gt_best_comb_lines = topk_lines[gt_best_comb_list]
        pred_best_comb_idx = outputs['pred_score'][0, :, 0].argmax(-1)
        pred_best_comb_list = outputs['comb_list'][0, pred_best_comb_idx]
        pred_best_comb_lines = topk_lines[pred_best_comb_list]

        results = {'topk_lines': topk_lines, 'gt_comb': gt_best_comb_lines, 'pred_comb': pred_best_comb_lines}

        gt_line = gt_pts[0]
        valid = gt_line[:, 0] != -1000
        gt_line = coord_unnorm(cfg, gt_line[valid])
        if (epoch % 10 == 0) and cfg.viz:
            viz_tools.viz_test(cfg, img[0], gt_line, results, img_name[0], epoch)

        if epoch >= cfg.start_eval_epoch:
            slines = pred_best_comb_lines.reshape(-1, 4)
            hiou += outputs['gt_hious'][0, pred_best_comb_idx]      # eval_line.measure_HIoU(slines, gt_line)
            match = eval_line.measure_PRF(slines, gt_line)
            EA_scores['pred'][img_name[0]] = slines
            EA_scores['gt'][img_name[0]] = gt_line
            match1['p'][img_name[0]] = match['p']
            match1['r'][img_name[0]] = match['r']
            max_hiou += outputs['gt_hious'][0, gt_best_comb_idx]

    max_HIoU = max_hiou / len(test_loader)
    if epoch == cfg.start_eval_epoch:
        print('The maximum preformance is %.5f' % max_HIoU)

    HIoU = hiou / len(test_loader)
    # AUC_P, AUC_R, AUC_F = eval_line.measure_AUC_PRF(match1)  # Use mIoU metric
    # print('%s ==> Total HIoU %5f / AUC_P %5f / AUC_R %5f / AUC_F %5f' % (cfg.dataset_name, HIoU, AUC_P, AUC_R, AUC_F))
    precision, recall, f_score = eval_line.measure_EA_score_dict(EA_scores)    # Use EA-score metric
    print('%s ==> Total HIoU %5f / P %5f / R %5f / F %5f' % (cfg.dataset_name, HIoU, precision, recall, f_score))

    best_HIoU = save_best_line_detection_model(cfg, model, optimizer, epoch, HIoU, best_HIoU)
    return best_HIoU


def analysis_line_detection(cfg, model, test_loader, criterion):
    model.eval()
    criterion.eval()

    eval_line = Evaluation_Semantic_Line(cfg)
    viz_tools = Visualizer(cfg)
    HIoU = 0
    N = 0
    match1 = {'p': {}, 'r': {}}
    EA_scores = {'pred': {}, 'gt': {}}

    pickle_results = {'out': {}}
    for i, batch in enumerate(test_loader):
        print('[%d/%d]' % (i, len(test_loader)), end='\r')

        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        for k, v in batch['detector_outputs'].items():
            batch['detector_outputs'][k] = v.cuda()

        targets = {'pts': gt_pts, 'params': gt_params}

        with torch.no_grad():
            outputs = model(img, batch['detector_outputs'], targets=targets)
            cross_attn = model.cross_attn[-1].cross_attn_weight

        gt_line = gt_pts[0]
        valid = gt_line[:, 0] != -1000
        gt_line = coord_unnorm(cfg, gt_line[valid])

        if cfg.viz:
            topk_lines = outputs[f'topk_line_pts'][0]
            gt_best_comb_idx = outputs['gt_score'][0, :].argmax(-1)
            gt_best_comb_list = outputs['comb_list'][0, gt_best_comb_idx]
            gt_best_comb_lines = topk_lines[gt_best_comb_list]
            pred_best_comb_idx = outputs['pred_score'][0, :, 0].argmax(-1)
            pred_best_comb_list = outputs['comb_list'][0, pred_best_comb_idx]
            pred_best_comb_lines = topk_lines[pred_best_comb_list]
            topk_best_comb_idx = torch.topk(outputs['pred_score'][0, :, 0], 5)[1]

            results = {'topk_lines': topk_lines, 'gt_comb': gt_best_comb_lines, 'pred_comb': pred_best_comb_lines,
                       'pred_score': outputs['pred_score'][0, :, 0], 'gt_score': outputs['gt_score'][0, :],
                       'gt_best_comb_idx': gt_best_comb_idx, 'pred_best_comb_idx': pred_best_comb_idx,
                       'comb_list': outputs['comb_list'][0]}

            viz_tools.viz_analysis(cfg, img[0], gt_line, results, img_name[0])

        topk_lines = outputs[f'topk_line_pts'][0]
        pred_best_comb_idx = outputs['pred_score'][0, :, 0].argmax(-1)
        pred_best_comb_list = outputs['comb_list'][0, pred_best_comb_idx]
        pred_best_comb_lines = topk_lines[pred_best_comb_list]
        slines = pred_best_comb_lines.reshape(-1, 4)

        slines = coord_rescale(cfg, slines)
        gt_line = coord_rescale(cfg, gt_line)

        hiou = eval_line.measure_HIoU(slines, gt_line)  # outputs['gt_hious'][0, pred_best_comb_idx]
        match = eval_line.measure_PRF(slines, gt_line)
        EA_scores['pred'][img_name[0]] = slines
        EA_scores['gt'][img_name[0]] = gt_line
        match1['p'][img_name[0]] = match['p']
        match1['r'][img_name[0]] = match['r']
        HIoU += hiou
        N += len(img)

        if cfg.save_pickle:
            pickle_results['out']['mul_reg'] = slines
            dir, name = os.path.split(img_path[0])
            save_pickle(dir_name=os.path.join(cfg.output_root, f'comparison/{cfg.dataset_name}/pickle'),
                        file_name=name[:-4],
                        data=pickle_results)

    HIoU = HIoU / N
    print('# of data : ', N)
    if 'SEL' in cfg.dataset_name:
        AUC_P, AUC_R, AUC_F = eval_line.measure_AUC_PRF(match1)                     # Use mIoU metric
        print('%s %04d ==> Total HIoU %5f / AUC_P %5f / AUC_R %5f / AUC_F %5f' % (cfg.dataset_name, len(test_loader), HIoU, AUC_P, AUC_R, AUC_F))
    else:
        precision, recall, f_score = eval_line.measure_EA_score_dict(EA_scores)     # Use EA-score metric
        print('%s %04d ==> Total HIoU %5f / P %5f / R %5f / F %5f' % (cfg.dataset_name, len(test_loader), HIoU, precision, recall, f_score))


def coord_rescale(cfg, src):
    # For fair comparison with existing methods
    dst = src / (cfg.image_size - 1) * (400 - 1)
    return dst


def save_pickle(dir_name, file_name, data):
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, file_name + '.pickle'), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
