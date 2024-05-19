from dataloaders.SEL.SEL_dataset import Train_Dataset_SEL, Test_Dataset_SEL
from dataloaders.SEL_Hard.SEL_Hard_dataset import Test_Dataset_SEL_Hard
from dataloaders.NKL.NKL_dataset import Train_Dataset_NKL, Test_Dataset_NKL
from dataloaders.CDL.CDL_dataset import Train_Dataset_CDL, Test_Dataset_CDL


def load_train_dataset(cfg):
    if 'SEL' in cfg.dataset_name:
        train_dataset = Train_Dataset_SEL(cfg)
    elif 'NKL' in cfg.dataset_name:
        train_dataset = Train_Dataset_NKL(cfg)
    else:
        train_dataset = Train_Dataset_CDL(cfg)

    print('# of train data : ', len(train_dataset))
    return train_dataset

def load_test_dataset(cfg):
    if 'SEL' in cfg.dataset_name:
        if 'Hard' in cfg.dataset_name:
            test_dataset = Test_Dataset_SEL_Hard(cfg)
        else:
            test_dataset = Test_Dataset_SEL(cfg)
    elif 'NKL' in cfg.dataset_name:
        test_dataset = Test_Dataset_NKL(cfg)
    else:
        test_dataset = Test_Dataset_CDL(cfg)

    print('# of test data : ', len(test_dataset))
    return test_dataset

