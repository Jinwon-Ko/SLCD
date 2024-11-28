import os
import torch

from torch.utils.data import DataLoader
from dataloaders.factory import load_train_dataset, load_test_dataset

from engines.train.train_process import train_line_detection_with_combination
from engines.test.test_process import test_line_detection_with_combination


def do_train_process(cfg, model, criterion, optimizer, lr_scheduler):
    model.cuda()
    criterion.cuda()

    metrics = []

    train_dataset = load_train_dataset(cfg)
    test_dataset = load_test_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    if len(os.listdir(cfg.pickle_dir)) != (len(train_dataset) + len(test_dataset)):
        from models.networks.line_detector.model import Detector
        from engines.train.train_process import forward_line_detector
        detector = Detector(cfg).cuda()
        checkpoint = torch.load(cfg.detector_ckpt)
        detector.load_state_dict(checkpoint['model_state_dict'], strict=False)
        checkpoint = None   # For memory
        forward_line_detector(cfg, detector)
        detector = None     # For memory

    best_HIoU = 0
    for epoch in range(cfg.epochs):
        model, optimizer = train_line_detection_with_combination(cfg, epoch, model, train_loader, criterion, optimizer)
        best_HIoU = test_line_detection_with_combination(cfg, epoch, model, test_loader, criterion, optimizer, best_HIoU)
        lr_scheduler.step(epoch)
        metrics.append(best_HIoU)

    print('best_HIoU : ', best_HIoU)
