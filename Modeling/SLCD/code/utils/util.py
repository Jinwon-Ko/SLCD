import os
import shutil
import random
import pickle
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch


def fix_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# logging
def get_current_time():
    _now = datetime.now().strftime('%m%d_%H%M%S')
    # _now = datetime.now()
    # _now = str(_now)[:-7]
    return _now


def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text),
        f.close()

# convertor
def to_tensor(data):
    try:
        return torch.from_numpy(data).cuda()
    except:
        torch.from_numpy(data)

def to_np(data):
    try:
        return data.cpu().numpy()
    except:
        return data.detach().cpu().numpy()


# directory & file
def mkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def rmfile(path):
    if os.path.exists(path):
        os.remove(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

# pickle
def save_pickle(dir_name, file_name, data):
    '''
    :param file_path: ...
    :param data:
    :return:
    '''
    mkdir(dir_name)
    with open(os.path.join(dir_name, file_name + '.pickle'), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
    with open(file_path + '.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

# functions
def get_pdefined_anchor(path):
    # get predefined anchors(x1, y1, x2, y2)
    pdefined_anchors = np.array(pickle.load(open(path, 'rb'), encoding='iso-8859-1')).astype(np.float32)
    print('num of pre-defined anchors: ', pdefined_anchors.shape)
    return pdefined_anchors


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


# Checkpoints
def load_model(args, net, optimizer=None, load_optim_params=False):
    checkpoint = torch.load(args.init_model, map_location=torch.device("cuda:%s" % (0) if torch.cuda.is_available() else "cpu"))
    model_dict = net.state_dict()

    new_model_state_dict = OrderedDict()
    for k, v in model_dict.items():
        #if k in checkpoint['model_state_dict'].keys():
        if k[7:] in checkpoint['model_state_dict'].keys():
            #new_model_state_dict[k] = checkpoint['model_state_dict'][k]
            new_model_state_dict[k] = checkpoint['model_state_dict'][k[7:]]
            #print(f'Loaded\t{k}')
        else:
            new_model_state_dict[k] = v
            print(f'Not Loaded\t{k}')
    net.load_state_dict(new_model_state_dict)

    print("=> loaded checkpoint '{}'".format(args.init_model))

    if load_optim_params == True:

        optimizer_dict = optimizer.state_dict()
        optimizer_dict.update(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(optimizer_dict)

        print("=> loaded optimizer params '{}'".format(args.init_model))


def save_final_model_line_detection(args, net, optimizer, epoch):
    save_dir = os.path.join(args.save_folder, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_dir + '/' + f'checkpoint_final_line_detection.pth'))


def save_best_partitioning_model(args, net, optimizer, epoch, loss, best_loss):
    save_dir = os.path.join(args.save_folder, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    if best_loss > loss:
        best_loss = loss

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir + '/' + f'checkpoint_best_image_partitioning.pth'))
        print('Saved best_model to ' + save_dir + '/' + f'checkpoint_best_image_partitioning.pth')
        print(f'Epoch [{epoch:03d}] Best model performances : ' + f'[Loss {loss:.5f}]')
        logger("Average Metrics : [Epoch: %d] [Loss %5f]\n" % (epoch, loss), f'{args.save_folder}/results_image_partitioning.txt')
    return best_loss


def save_best_recall_model(args, net, optimizer, epoch, recall, best_recall):
    save_dir = os.path.join(args.save_folder, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    if recall > best_recall:
        best_recall = recall

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir + '/' + f'checkpoint_best_recall.pth'))
        print('Saved best_model to ' + save_dir + '/' + f'checkpoint_best_recall.pth')
        print(f'Epoch [{epoch:03d}] Best model performances : ' + f'[Recall {recall:.5f}]')
        logger("Average Metrics : [Epoch: %d] [Recall %5f]\n" % (epoch, recall), f'{args.save_folder}/results_line_detection.txt')
    return best_recall


def save_best_line_detection_model(args, net, optimizer, epoch, HIoU, best_HIoU):
    save_dir = os.path.join(args.save_folder, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    if HIoU > best_HIoU:
        best_HIoU = HIoU

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir + '/' + f'checkpoint_best_line_detection.pth'))
        print('Saved best_model to ' + save_dir + '/' + f'checkpoint_best_line_detection.pth')
        print(f'Epoch [{epoch:03d}] Best model performances : ' + f'[HIoU {HIoU:.5f}]')
        logger("Average Metrics : [Epoch: %d] [HIoU %5f]\n" % (epoch, HIoU), f'{args.save_folder}/results_line_detection.txt')
    return best_HIoU


def copy_code(args):
    if os.path.exists(os.path.join(args.save_folder, 'code')):
        shutil.rmtree(os.path.join(args.save_folder, 'code'))
    os.makedirs(os.path.join(args.save_folder, 'train/configs'), exist_ok=True)
    shutil.copytree(os.path.join(args.proj_root),
                    os.path.join(args.save_folder, 'code'))
