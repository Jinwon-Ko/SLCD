import time
import torch

from utils.util import to_np, save_best_recall_model
from utils.misc import coord_norm, coord_unnorm
from utils.viz_utils import Visualizer
from utils.line_utils import convert_to_line, NMS
from utils.calculate_metrics import Evaluation_Semantic_Line


def test_line_detection(cfg, epoch, model, test_loader, criterion, optimizer, best_recall):
    if not epoch >= cfg.start_eval_epoch:
        return best_recall
    model.eval()
    criterion.eval()

    eval_line = Evaluation_Semantic_Line(cfg)
    viz_tools = Visualizer(cfg)
    metric_dict = get_metrics()
    for i, batch in enumerate(test_loader):
        print('[Epoch: %d][%d/%d]' % (epoch, i, len(test_loader)), end='\r')

        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        targets = {'pts': gt_pts, 'params': gt_params}

        gt_line = gt_pts[0]
        valid = gt_line[:, 0] != -1000
        gt_line = coord_unnorm(cfg, gt_line[valid])

        with torch.no_grad():
            outputs = model(img=img)
            comb_outputs = model.prepare_for_combination(outputs)

        probas = outputs['prob'][0, :, 1]
        line_pts = outputs['pts'][0]
        keep = NMS(cfg, line_pts, probas, threshold=cfg.test_nms_threshold)

        topk_prob = comb_outputs['topk_prob'][0]
        topk_line_pts = comb_outputs['topk_line_pts'][0]

        results = {'probas': probas, 'lines': line_pts, 'keep': keep,
                   'topk_line_pts': topk_line_pts, 'topk_prob': topk_prob}

        if epoch % 10 == 0:
            viz_tools.viz_test(cfg, img[0], gt_line, results, img_name[0], epoch)
        if epoch >= cfg.start_eval_epoch:
            metric_dict = calculate_metrics(metric_dict, results, gt_line, eval_line, img_name[0])

    HIoU = metric_dict['HIoU'] / len(test_loader)
    precision, recall, f_score = eval_line.measure_EA_score_dict(metric_dict['EA_score'])  # Use EA-score metric
    print('%s ==> Total HIoU %5f / P %5f / R %5f / F %5f' % (cfg.dataset_name, HIoU, precision, recall, f_score))

    best_recall = save_best_recall_model(cfg, model, optimizer, epoch, recall, best_recall)
    return best_recall


def analysis_line_detection(cfg, model, test_loader, criterion):
    model.eval()
    criterion.eval()

    eval_line = Evaluation_Semantic_Line(cfg)
    viz_tools = Visualizer(cfg)
    metric_dict = get_metrics()
    for i, batch in enumerate(test_loader):
        print('[%d/%d]' % (i, len(test_loader)), end='\r')

        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        targets = {'pts': gt_pts, 'params': gt_params}

        gt_line = gt_pts[0]
        valid = gt_line[:, 0] != -1000
        gt_line = coord_unnorm(cfg, gt_line[valid])

        with torch.no_grad():
            outputs = model(img=img)
            comb_outputs = model.prepare_for_combination(outputs)

        probas = outputs['prob'][0, :, 1]
        line_pts = outputs['pts'][0]
        keep = NMS(cfg, line_pts, probas, threshold=cfg.test_nms_threshold)
        topk_prob = comb_outputs['topk_prob'][0]
        topk_line_pts = comb_outputs['topk_line_pts'][0]

        results = {'probas': probas, 'lines': line_pts, 'keep': keep,
                   'topk_line_pts': topk_line_pts, 'topk_prob': topk_prob}

        viz_tools.viz_analysis(cfg, img[0], gt_line, results, img_name[0])
        metric_dict = calculate_metrics(metric_dict, results, gt_line, eval_line, img_name[0])

    HIoU = metric_dict['HIoU'] / len(test_loader)
    precision, recall, f_score = eval_line.measure_EA_score_dict(metric_dict['EA_score'])  # Use EA-score metric
    print('%s ==> Total HIoU %5f / P %5f / R %5f / F %5f' % (cfg.dataset_name, HIoU, precision, recall, f_score))


def get_metrics():
    metric_dict = {'HIoU': 0,
                   'AUC': {'p': {}, 'r': {}},
                   'EA_score': {'pred': {}, 'gt': {}}}
    return metric_dict


def calculate_metrics(metric_dict, result, gt, eval_line, img_name):
    slines = result['lines'][result['keep']].reshape(-1, 4)
    topk_line_pts = result['topk_line_pts'].reshape(-1, 4)
    match = eval_line.measure_PRF(slines, gt)

    metric_dict['HIoU'] += eval_line.measure_HIoU(slines, gt)
    metric_dict['EA_score']['pred'][img_name] = topk_line_pts
    metric_dict['EA_score']['gt'][img_name] = gt
    metric_dict['AUC']['p'][img_name] = match['p']
    metric_dict['AUC']['r'][img_name] = match['r']
    return metric_dict

