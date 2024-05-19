from torch.utils.data import DataLoader
from dataloaders.factory import load_train_dataset, load_test_dataset
from engines.train.train_process import train_line_detection
from engines.test.test_process import test_line_detection



def do_train_process(cfg, model, criterion, optimizer, lr_scheduler):
    model.cuda()
    criterion.cuda()

    metrics = []

    train_dataset = load_train_dataset(cfg)
    test_dataset = load_test_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    best_recall = 0
    for epoch in range(cfg.epochs):
        model, optimizer = train_line_detection(cfg, epoch, model, train_loader, criterion, optimizer)
        best_recall = test_line_detection(cfg, epoch, model, test_loader, criterion, optimizer, best_recall)
        lr_scheduler.step(epoch)
        metrics.append(best_recall)
    print('best_Recall : ', best_recall)
