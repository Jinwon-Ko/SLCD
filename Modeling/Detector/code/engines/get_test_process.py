import torch

from torch.utils.data import DataLoader
from dataloaders.factory import load_train_dataset, load_test_dataset
from engines.test.test_process import analysis_line_detection


def do_test_process(cfg, model, criterion):
    checkpoint = torch.load(cfg.init_model)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    checkpoint = None

    model.cuda()
    criterion.cuda()

    test_dataset = load_test_dataset(cfg)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    analysis_line_detection(cfg, model, test_loader,  criterion)