import torch
from utils.util import to_np, logger, save_final_model_line_detection
from utils.misc import coord_norm, coord_unnorm
from utils.viz_utils import Visualizer


def train_line_detection(cfg, epoch, model, train_loader, criterion, optimizer):
    print('Epoch %03d ========================================================' % epoch)
    model.train()
    criterion.train()

    viz_tools = Visualizer(cfg)
    loss_t = {'Total': 0, 'Cls': 0, 'Reg': 0}
    num = 0

    for i, batch in enumerate(train_loader):
        # load data
        img = batch['img'].cuda()
        img_path = batch['img_path']
        img_name = batch['img_name']
        gt_pts = batch['gt_lines'].cuda()
        gt_params = batch['gt_params'].cuda()

        targets = {'pts': gt_pts, 'params': gt_params}
        outputs = model(img=img, targets=targets)

        loss_dict = {}
        loss_dict.update(criterion.get_loss_for_line_detection(outputs, targets))
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            for b_idx in range(img.shape[0]):
                gt_line = gt_pts[b_idx]
                valid = gt_line[:, 0] != -1000
                gt_line = coord_unnorm(cfg, gt_line[valid])
                with torch.no_grad():
                    comb_outputs = model.prepare_for_combination(outputs)
                    outputs.update(comb_outputs)
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
