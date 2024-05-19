import math

import torch
import torch.optim as optim

def get_optimizer(cfg, net):
    if cfg.optim == 'AdamW':
        optimizer = optim.AdamW(params=net.parameters(),
                                lr=cfg.optim['lr'],
                                weight_decay=cfg.optim['weight_decay'],
                                betas=cfg.optim['betas'], eps=cfg.optim['eps'])

    elif cfg.optim == 'Adam_layerwise':
        optimizer = optim.Adam(
            [
                {"params": net.backbone.parameters(), "lr": cfg.backbone_lr}
            ], lr=cfg.lr, weight_decay=cfg.weight_decay)

    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg.scheduler == 'cosinewarmup':
        warm_up_epochs = 5
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
                math.cos((epoch - warm_up_epochs) / (cfg.epoch - warm_up_epochs) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    elif cfg.scheduler == 'warmup':
        warm_up_epochs = 5
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else cfg.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)

    return lr_scheduler

