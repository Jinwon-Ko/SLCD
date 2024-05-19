import os
from utils.util import get_current_time


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


class Config:
    def __init__(self):
        # path
        self.root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.dataset_name = 'CDL'
        self.gpu = '0'

        self.model_name = 'end2end_'

        self.viz = True
        self.run_mode = 'test_paper'     # ['train', 'test', 'test_paper']

        self.settings_for_dataset()
        self.settings_for_backbone()
        self.settings_for_network()
        self.settings_for_matcher()
        self.settings_for_training()
        self.settings_for_testing()
        self.settings_for_tuning()

        self.settings_for_path()

    def settings_for_tuning(self):
        self.perpendicular_rotate = False
        self.random_rotate = True

        self.num_region_queries = 8         # number of region queries
        self.num_anchors = 1024              # number of line anchors
        self.topk = 8

        # settings exp name
        self.exp = f'MS_agg_mq{self.num_region_queries}_nq{self.num_anchors}'
        if self.perpendicular_rotate or self.random_rotate:
            self.exp += '_w_rotaug'
        else:
            self.exp += '_wo_rotaug'

    def settings_for_dataset(self):
        self.dataset_root = os.path.abspath(os.path.join(self.root, '../..', f'Datasets/{self.dataset_name}/'))
        self.image_size = 480   # 320   384
        self.batch_size = 4
        self.num_workers = 4

        self.height = self.image_size
        self.width = self.image_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.max_theta = 90
        self.max_radius = self.image_size // 2
        if 'SEL' in self.dataset_name:
            self.n_max = 6
        else:
            self.n_max = 8

    def settings_for_backbone(self):
        self.backbone = 'resnet50'
        self.train_backbone = True
        self.pretrained = True          # Backbone Pretrained
        self.dim_feedforward = 2048

    def settings_for_network(self):
        self.hidden_dim = 96
        self.query_dim = 2
        self.nheads = 6
        self.dropout = 0.0
        self.gaussian_sigma = 20

        self.n_enc_layers = 3
        self.n_dec_layers = 3

    def settings_for_matcher(self):
        self.set_cost_class = 0
        self.set_cost_line = 1
        self.focal_alpha = 0.25

    def settings_for_training(self):
        self.epochs = 300
        self.optim = 'AdamW'
        self.scheduler = 'MultiStepLR'
        self.backbone_lr = 1e-4
        self.lr = 1e-4
        self.weight_decay = 0    # 1e-4

        self.optim = dict()
        self.optim['weight_decay'] = 1e-4
        self.optim['gamma'] = 0.5
        self.optim['betas'] = (0.9, 0.999)
        self.optim['eps'] = 1e-8
        self.optim['mode'] = 'adam_w'

        self.milestones = [30, 60, 90, 120, 150]
        self.gamma = 0.5

        self.loss_cls_coef = 1
        self.loss_reg_coef = 5

        self.threshold_theta = 2.5
        self.threshold_radius = 4

    def settings_for_testing(self):
        self.start_eval_epoch = 1
        self.prob_threshold = 0.5
        self.test_nms_threshold = 0.01

    def settings_for_path(self):
        self.code_name = os.getcwd().split('/')[-1]
        self.output_name = self.code_name.replace('code', 'output')
        self.proj_root = os.path.join(self.root, f'{self.code_name}')
        self.output_root = os.path.join(self.root, f'{self.output_name}', self.exp)

        self.exp_name = f'{self.model_name}{self.dataset_name}'
        self.viz_dir = os.path.join(self.output_root, f'display/{self.exp_name}')
        self.save_folder = os.path.join(self.output_root, f'weights/{self.exp_name}')
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.save_folder, exist_ok=True)

        if self.run_mode == 'test':
            detector_path = 'ckpt/checkpoint_best_recall.pth'
            self.init_model = os.path.join(self.save_folder, detector_path)

        elif self.run_mode == 'test_paper':
            detector_path = f'ckpt/checkpoint_paper_{self.dataset_name}.pth'
            self.init_model = os.path.join(self.root, f'{self.output_name}', detector_path)

        self.now = get_current_time()
        self.log_configs()

    def log_configs(self, log_file='log.txt'):
        if os.path.exists(f'{self.save_folder}/{log_file}'):
            log_file = open(f'{self.save_folder}/{log_file}', 'a')
        else:
            log_file = open(f'{self.save_folder}/{log_file}', 'w')

        write_log(log_file, '------------ Options -------------')
        for k in vars(self):
            write_log(log_file, f'{str(k)}: {str(vars(self)[k])}')
        write_log(log_file, '-------------- End ----------------')

        log_file.close()

        return

if __name__ == "__main__":
    c = Config()
    print('debug... ')
