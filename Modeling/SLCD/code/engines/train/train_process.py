import torch
from torch.utils.data import DataLoader

from models.tools.network_tools import Make_Comb
from dataloaders.factory import load_train_dataset, load_test_dataset
from utils.util import to_np, logger, save_final_model_line_detection
from utils.misc import coord_norm, coord_unnorm
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluation_Semantic_Line


def forward_line_detector(cfg, model):
    print('Forward line detector ===========================================')
    model.eval()

    eval_line = Evaluation_Semantic_Line(cfg)
    viz_tools = Visualizer(cfg)
    comb_generator = Make_Comb(cfg, scale_factor=model.scale_factor)
    # ======================================================================
    # Forward for test dataset
    match1 = {'p': {}, 'r': {}}
    test_dataset = load_test_dataset(cfg, forward_detctor=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    for i, batch in enumerate(test_loader):
        print('[%d/%d]' % (i, len(test_loader)), end='\r')

        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        gt_line = gt_pts[0]
        valid = gt_line[:, 0] != -1000
        gt_line = coord_unnorm(cfg, gt_line[valid])
        gt_param = gt_params[0][valid]

        targets = {'pts': gt_line, 'params': gt_param}

        with torch.no_grad():
            outputs = model(img=img)
            comb_outputs = model.prepare_for_combination(outputs)

        probas = comb_outputs['topk_prob'][0]
        line_params = comb_outputs['topk_line_params'][0]
        line_pts = comb_outputs['topk_line_pts'][0]

        results = {'topk_line_pts': line_pts, 'topk_line_params': line_params, 'topk_prob': probas}
        # results = comb_generator.check_results(results, targets)    --> Do not process on test dataset
        match = comb_generator.save_forwarding_results(results, targets, img_name[0])
        match1['p'][img_name[0]] = match['p']
        match1['r'][img_name[0]] = match['r']
        viz_tools.viz_forward_detector(cfg, img[0], gt_line, results['topk_line_pts'], img_name[0], dataset='test')

    AUC_P, AUC_R, AUC_F = eval_line.measure_AUC_PRF(match1)
    print('Forward Detector on %s Testset ==> AUC_P %5f / AUC_R %5f / AUC_F %5f' % (cfg.dataset_name, AUC_P, AUC_R, AUC_F))

    # ======================================================================
    # Forward for train dataset
    if 'Hard' not in cfg.dataset_name:
        match1 = {'p': {}, 'r': {}}
        train_dataset = load_train_dataset(cfg, forward_detctor=True)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
        for i, batch in enumerate(train_loader):
            print('[%d/%d]' % (i, len(train_loader)), end='\r')

            # load data
            img = batch['img'].cuda()
            img_path = batch['img_path']
            img_name = batch['img_name']
            gt_pts = batch['gt_lines'].cuda()
            gt_params = batch['gt_params'].cuda()

            gt_line = gt_pts[0]
            valid = gt_line[:, 0] != -1000
            gt_line = coord_unnorm(cfg, gt_line[valid])
            gt_param = gt_params[0][valid]

            targets = {'pts': gt_line, 'params': gt_param}

            with torch.no_grad():
                outputs = model(img=img)
                comb_outputs = model.prepare_for_combination(outputs)

            probas = comb_outputs['topk_prob'][0]
            line_params = comb_outputs['topk_line_params'][0]
            line_pts = comb_outputs['topk_line_pts'][0]

            results = {'topk_line_pts': line_pts, 'topk_line_params': line_params, 'topk_prob': probas}
            results = comb_generator.check_results(results, targets)
            match = comb_generator.save_forwarding_results(results, targets, img_name[0])
            match1['p'][img_name[0]] = match['p']
            match1['r'][img_name[0]] = match['r']
            viz_tools.viz_forward_detector(cfg, img[0], gt_line, results['topk_line_pts'], img_name[0], dataset='train')

        AUC_P, AUC_R, AUC_F = eval_line.measure_AUC_PRF(match1)
        print('Forward Detector on %s Trainset ==> AUC_P %5f / AUC_R %5f / AUC_F %5f' % (cfg.dataset_name, AUC_P, AUC_R, AUC_F))


def train_line_detection_with_combination(cfg, epoch, model, train_loader, criterion, optimizer):
    print('Epoch %03d ========================================================' % epoch)
    model.train()
    criterion.train()

    viz_tools = Visualizer(cfg)
    loss_t = {'Total': 0, 'KLD': 0, 'Reg': 0, 'Ranking': 0}
    num = 0

    for i, batch in enumerate(train_loader):
        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        for k, v in batch['detector_outputs'].items():
            batch['detector_outputs'][k] = v.cuda()

        targets = {'pts': gt_pts, 'params': gt_params}
        outputs = model(img, batch['detector_outputs'], targets=targets)

        loss_dict = {}
        loss_dict.update(criterion.get_KLDloss_each_pairs(outputs, targets))
        loss_dict.update(criterion.score_weighted_regression_loss(outputs['pred_score'][..., 0], outputs['gt_score']))
        loss_dict.update(criterion.score_rank_loss(outputs['pred_score'][..., 0], outputs['gt_score']))
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if cfg.viz and (i % 10 == 0):
            for b_idx in range(img.shape[0]):
                gt_line = gt_pts[b_idx]
                valid = gt_line[:, 0] != -1000
                gt_line = coord_unnorm(cfg, gt_line[valid])
                viz_tools.viz_train(cfg, img[b_idx], outputs, gt_line, targets, criterion, b_idx, num=num + b_idx)

        num += img.shape[0]
        loss_t, log = make_batch_logging(loss_t, loss_dict)
        loss_t['Total'] += losses.item()
        txt = 'Total: %.5f  %s' % (losses.item(), log)
        print('[Epoch %d][%d/%d][Losses %s]' % (epoch, i, len(train_loader), txt), end='\r')

    loss_t, log = make_logging(loss_t, num)

    # logging
    logger("[Epoch %d Average Losses] %s\n" % (epoch, log), f'{cfg.save_folder}/losses_line_detection.txt')
    print('\n[Epoch %d Average Losses] %s' % (epoch, log))
    save_final_model_line_detection(cfg, model, optimizer, epoch)
    return model, optimizer


def make_batch_logging(loss_t, loss_dict):
    txt = ''
    for key, value in loss_dict.items():
        loss_t[key] += value.item()
        txt += f'{key}: {value.item():.5f}  '
    return loss_t, txt


def make_logging(loss_t, denom):
    txt = ''
    for key, value in loss_t.items():
        loss_t[key] = value / denom
        txt += f'[{key}: {loss_t[key]:.5f}] '
    return loss_t, txt
