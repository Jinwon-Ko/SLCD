from dataloaders.SEL.SEL_dataset import Train_Dataset_SEL, Test_Dataset_SEL, Train_Dataset_SEL_combination, Test_Dataset_SEL_combination
from dataloaders.SEL_Hard.SEL_Hard_dataset import Test_Dataset_SEL_Hard, Test_Dataset_SEL_Hard_combination
from dataloaders.NKL.NKL_dataset import Train_Dataset_NKL, Test_Dataset_NKL, Train_Dataset_NKL_combination, Test_Dataset_NKL_combination
from dataloaders.CDL.CDL_dataset import Train_Dataset_CDL, Test_Dataset_CDL, Train_Dataset_CDL_combination, Test_Dataset_CDL_combination


def load_train_dataset(cfg, forward_detctor=False):
    if 'SEL' in cfg.dataset_name:
        train_dataset = Train_Dataset_SEL(cfg)
        if 'combination' in cfg.model_name:
            if forward_detctor:
                cfg.perpendicular_rotate = False
                cfg.random_rotate = False
                train_dataset = Train_Dataset_SEL(cfg, flip=False)
            else:
                train_dataset = Train_Dataset_SEL_combination(cfg)

    elif 'NKL' in cfg.dataset_name:
        train_dataset = Train_Dataset_NKL(cfg)
        if 'combination' in cfg.model_name:
            if forward_detctor:
                cfg.perpendicular_rotate = False
                cfg.random_rotate = False
                train_dataset = Train_Dataset_NKL(cfg, flip=False)
            else:
                train_dataset = Train_Dataset_NKL_combination(cfg)

    else:
        train_dataset = Train_Dataset_CDL(cfg)
        if 'combination' in cfg.model_name:
            if forward_detctor:
                cfg.perpendicular_rotate = False
                cfg.random_rotate = False
                train_dataset = Train_Dataset_CDL(cfg, flip=False)
            else:
                train_dataset = Train_Dataset_CDL_combination(cfg)

    return train_dataset


def load_test_dataset(cfg, forward_detctor=False):
    if 'SEL' in cfg.dataset_name:
        if 'Hard' in cfg.dataset_name:
            test_dataset = Test_Dataset_SEL_Hard(cfg)
            if 'combination' in cfg.model_name:
                if not forward_detctor:
                    test_dataset = Test_Dataset_SEL_Hard_combination(cfg)
        else:
            test_dataset = Test_Dataset_SEL(cfg)
            if 'combination' in cfg.model_name:
                if not forward_detctor:
                    test_dataset = Test_Dataset_SEL_combination(cfg)

    elif 'NKL' in cfg.dataset_name:
        test_dataset = Test_Dataset_NKL(cfg)
        if 'combination' in cfg.model_name:
            if not forward_detctor:
                test_dataset = Test_Dataset_NKL_combination(cfg)    # Train_Dataset_NKL_combination

    elif 'CDL' in cfg.dataset_name:
        test_dataset = Test_Dataset_CDL(cfg)
        if 'combination' in cfg.model_name:
            if not forward_detctor:
                test_dataset = Test_Dataset_CDL_combination(cfg)

    return test_dataset

