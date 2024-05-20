import os
import torch

from torch.utils.data import DataLoader
from dataloaders.factory import load_test_dataset

from engines.train.train_process import forward_line_detector
from engines.test.test_process import analysis_line_detection
from models.networks.line_detector.model import Detector


def do_test_process(cfg, model, criterion):
    if len(os.listdir(cfg.pickle_dir)) == 0:
        detector = Detector(cfg).cuda()
        checkpoint = torch.load(cfg.detector_ckpt)
        detector.load_state_dict(checkpoint['model_state_dict'], strict=False)
        checkpoint = None  # For memory
        forward_line_detector(cfg, detector)
        detector = None  # For memory

    checkpoint = torch.load(cfg.init_model)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    checkpoint = None

    model.cuda()
    criterion.cuda()

    test_dataset = load_test_dataset(cfg)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    analysis_line_detection(cfg, model, test_loader, criterion)
