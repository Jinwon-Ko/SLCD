import torch.nn as nn
from utils.util import *

from sklearn.metrics import auc

# For Evaluate Metrics

# For Precision / Recall
class Line(object):
    def __init__(self, coordinates=[0, 0, 1, 1]):
        """
        coordinates: [y0, x0, y1, x1]
        """
        assert isinstance(coordinates, list)
        assert len(coordinates) == 4
        assert coordinates[0] != coordinates[2] or coordinates[1] != coordinates[3]
        self.__coordinates = coordinates

    @property
    def coord(self):
        return self.__coordinates

    @property
    def length(self):
        start = np.array(self.coord[:2])
        end = np.array(self.coord[2::])
        return np.sqrt(((start - end) ** 2).sum())

    def angle(self):
        y0, x0, y1, x1 = self.coord
        if x0 == x1:
            return -np.pi / 2
        return np.arctan((y0 - y1) / (x0 - x1))

    def rescale(self, rh, rw):
        coor = np.array(self.__coordinates)
        r = np.array([rh, rw, rh, rw])
        self.__coordinates = np.round(coor * r).astype(np.int).tolist()

    def __repr__(self):
        return str(self.coord)

def sa_metric(angle_p, angle_g):
    d_angle = np.abs(angle_p - angle_g)
    d_angle = min(d_angle, np.pi - d_angle)
    d_angle = d_angle * 2 / np.pi
    return max(0, (1 - d_angle)) ** 2

def se_metric(coord_p, coord_g, size=(400, 400)):
    c_p = [(coord_p[0] + coord_p[2]) / 2, (coord_p[1] + coord_p[3]) / 2]
    c_g = [(coord_g[0] + coord_g[2]) / 2, (coord_g[1] + coord_g[3]) / 2]
    d_coord = np.abs(c_p[0] - c_g[0])**2 + np.abs(c_p[1] - c_g[1])**2
    d_coord = np.sqrt(d_coord) / max(size[0], size[1])
    return max(0, (1 - d_coord)) ** 2

def EA_metric(l_pred, l_gt, size=(400, 400)):
    se = se_metric(l_pred.coord, l_gt.coord, size=size)
    sa = sa_metric(l_pred.angle(), l_gt.angle())
    return sa * se

def caculate_precision(b_points, gt_coords, thresh=0.90):
    N = len(b_points)
    if N == 0:
        return 0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_p in enumerate(b_points):
        if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
            continue
        l_pred = Line(list(coord_p))
        for coord_g in gt_coords:
            l_gt = Line(list(coord_g))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

def caculate_recall(b_points, gt_coords, thresh=0.90):
    N = len(gt_coords)
    if N == 0:
        return 1.0, 0
    ea = np.zeros(N, dtype=np.float32)
    for i, coord_g in enumerate(gt_coords):
        l_gt = Line(list(coord_g))
        for coord_p in b_points:
            if coord_p[0]==coord_p[2] and coord_p[1]==coord_p[3]:
                continue
            l_pred = Line(list(coord_p))
            ea[i] = max(ea[i], EA_metric(l_pred, l_gt))
    return (ea >= thresh).sum(), N

# For HIoU
class Evaluation_HIoU(nn.Module):
    def __init__(self, cfg):
        super(Evaluation_HIoU, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = 400   # cfg.height, 400
        self.width = 400    # cfg.width, 400
        self.area = self.height * self.width

        # candidates
        self.line_num = np.zeros(2)

        # generate grid
        self.grid = {}
        self.X = {}
        self.Y = {}
        self.generate_grid(1, self.height, self.width)

    def generate_grid(self, sf, height, width):
        X, Y = np.meshgrid(np.linspace(0, width - 1, width),
                           np.linspace(0, height - 1, height))

        self.X[sf] = torch.tensor(X, dtype=torch.float, requires_grad=False).cuda()
        self.Y[sf] = torch.tensor(Y, dtype=torch.float, requires_grad=False).cuda()
        self.grid[sf] = torch.cat((self.X[sf].view(1, height, width, 1),
                                   self.Y[sf].view(1, height, width, 1)), dim=3)

    def line_equation(self, mode=None):
        if mode == 'norm':
            data = self.line_pts_norm.clone()
        else:
            data = self.line_pts.clone()
            data = data.cuda().type(torch.float32)

        # data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
        line_eq = torch.zeros((data.shape[0], 3)).cuda()
        line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
        line_eq[:, 1] = -1
        line_eq[:, 2] = (-1 * line_eq[:, 0] * data[:, 0] + data[:, 1])
        check = ((data[:, 0] - data[:, 2]) == 0)

        return line_eq, check

    def calculate_distance(self, line_eq, check):  # line-point distance
        num = line_eq.shape[0]
        a = line_eq[:, 0].view(num, 1, 1)
        b = line_eq[:, 1].view(num, 1, 1)
        c = line_eq[:, 2].view(num, 1, 1)

        dist = (self.grid[self.sf][:, :, :, 0] * a + self.grid[self.sf][:, :, :, 1] * b + c) / \
               torch.sqrt(a * a + b * b)

        if True in check:
            dist[check == True] = (self.grid[self.sf][:, :, :, 0].cuda().type(torch.float32) -
                                   self.line_pts[check == True, 0].view(-1, 1, 1).cuda().type(torch.float32))
        self.dist_map = dist

    def generate_region_mask(self):
        b, h, w = self.dist_map.shape
        region1 = (0 < self.dist_map).view(b, h, w, 1)
        region2 = (self.dist_map < 0).view(b, h, w, 1)

        return torch.cat((region1, region2), dim=3)

    def update(self, line_pts, scale_factor):

        self.line_pts = line_pts
        self.sf = scale_factor

    def update_dataset_name(self, dataset_name):

        self.dataset_name = dataset_name

    def preprocess(self, line_pts):
        output = {'region_mask': {},
                  'line_mask': {},
                  'grid': {},
                  'weight': {}}

        self.update(line_pts, 1)
        line_eq, check = self.line_equation()
        self.calculate_distance(line_eq, check)
        region_mask = self.generate_region_mask()

        output['region_mask'] = region_mask

        return output

    def measure_IoU(self, X1, X2):
        X = X1 + X2

        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

        iou = X_inter / X_uni

        return iou

    def measure_hiou_metric(self, pred, gt):

        m = len(pred)
        n = len(gt)
        score_table = torch.zeros((m, n), dtype=torch.float).cuda()

        for i in range(m):
            score_table[i, :] = self.measure_IoU(pred[i:i+1], gt)

        result = 0

        for i in range(m):
            result += score_table[i].max()
        score_table_T = score_table.T
        for j in range(n):
            result += score_table_T[j].max()
        result = result / (m + n)

        return result

    def generate_inter_region_mask(self, region_mask):
        memory_region_mask = region_mask[:1].clone()

        for i in range(1, region_mask.shape[0]):
            temp_region_mask = torch.BoolTensor([]).cuda()
            for j in range(2):
                region = memory_region_mask * region_mask[i:i+1, :, :, j:j+1]
                temp_region_mask = torch.cat((temp_region_mask, region), dim=3)
            memory_region_mask = temp_region_mask.clone()

        memory_region_mask = memory_region_mask[0].permute(2, 0, 1)
        area = torch.sum(memory_region_mask, dim=(1, 2))
        idx = (area != 0).nonzero()[:, 0]
        piece_mask = memory_region_mask[idx]

        return piece_mask.type(torch.float)

    def generate_inter_region_mask_detector(self, region_mask):
        memory_region_mask = region_mask[:1].clone()

        for i in range(1, region_mask.shape[0]):
            temp_region_mask = torch.BoolTensor([]).cuda()
            for j in range(2):
                region = memory_region_mask * region_mask[i:i+1, :, :, j:j+1]
                temp_region_mask = torch.cat((temp_region_mask, region), dim=3)
            memory_region_mask = temp_region_mask.clone()

        memory_region_mask = memory_region_mask[0].permute(2, 0, 1)
        area = torch.sum(memory_region_mask, dim=(1, 2))
        idx = (area > self.area * 0.001).nonzero()[:, 0]           ###
        piece_mask = memory_region_mask[idx]

        return piece_mask.type(torch.float)

    def run(self, pred_pts, gt_pts):

        result = dict()

        if pred_pts.shape[0] != 0:
            pred_mask = self.preprocess(pred_pts)
            gt_mask = self.preprocess(gt_pts)
            self.line_num[:] = pred_pts.shape[0], gt_pts.shape[0]
            pred_inter_mask = self.generate_inter_region_mask(pred_mask['region_mask'])
            gt_inter_mask = self.generate_inter_region_mask(gt_mask['region_mask'])

            result['IOU'] = self.measure_hiou_metric(pred_inter_mask, gt_inter_mask)

        else:
            result['IOU'] = "not_detection_image"

        return result

    def forward_Detector(self, pred_pts, gt_pts):

        result = dict()

        if pred_pts.shape[0] != 0:
            pred_mask = self.preprocess(pred_pts)
            gt_mask = self.preprocess(gt_pts)
            self.line_num[:] = pred_pts.shape[0], gt_pts.shape[0]
            pred_inter_mask = self.generate_inter_region_mask_detector(pred_mask['region_mask'])
            gt_inter_mask = self.generate_inter_region_mask_detector(gt_mask['region_mask'])

            result['IOU'] = self.measure_hiou_metric(pred_inter_mask, gt_inter_mask)

        else:
            result['IOU'] = "not_detection_image"

        return result

def divided_region_mask(line_pts, size):

    line_num, _ = line_pts.shape
    width, height = int(size[0]), int(size[1])

    X, Y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))  # after x before
    X = torch.tensor(X, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()
    Y = torch.tensor(Y, dtype=torch.float, requires_grad=False).unsqueeze(0).cuda()

    check = ((line_pts[:, 0] - line_pts[:, 2]) == 0).type(torch.float)

    mask1 = torch.zeros((line_num, height, width), dtype=torch.float32).cuda()
    mask2 = torch.zeros((line_num, height, width), dtype = torch.float32).cuda()

    mask1[check == 1, :, :] = (X < line_pts[:, 0].view(line_num, 1, 1)).type(torch.float)[check == 1, :, :]
    mask2[check == 1, :, :] = (X >= line_pts[:, 0].view(line_num, 1, 1)).type(torch.float)[check == 1, :, :]

    a = (line_pts[:, 1] - line_pts[:, 3]) / (line_pts[:, 0] - line_pts[:, 2])
    b = -1 * a * line_pts[:, 0] + line_pts[:, 1]

    a = a.view(line_num, 1, 1)
    b = b.view(line_num, 1, 1)

    mask1[check == 0, :, :] = (Y < a * X + b).type(torch.float32)[check == 0, :, :]
    mask2[check == 0, :, :] = (Y >= a * X + b).type(torch.float32)[check == 0, :, :]

    return torch.cat((mask1.unsqueeze(1), mask2.unsqueeze(1)), dim=1)

def measure_IoU_set(ref_mask, tar_mask):
    ref_num = ref_mask.shape[0]
    tar_num = tar_mask.shape[0]

    miou = torch.zeros((tar_num, ref_num), dtype=torch.float32).cuda()

    for i in range(tar_num):
        iou_1, check1 = measure_IoU(tar_mask[i, 0].unsqueeze(0), ref_mask[:, 0])
        iou_2, check2 = measure_IoU(tar_mask[i, 1].unsqueeze(0), ref_mask[:, 1])

        check = (check1 * check2).type(torch.float32)
        max_check = (miou[i] < check * (iou_1 + iou_2) / 2).type(torch.float32)
        miou[i][max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

        iou_1, check1 = measure_IoU(tar_mask[i, 1].unsqueeze(0), ref_mask[:, 0])
        iou_2, check2 = measure_IoU(tar_mask[i, 0].unsqueeze(0), ref_mask[:, 1])

        check = (check1 * check2).type(torch.float32)
        max_check = (miou[i] < check * (iou_1 + iou_2) / 2).type(torch.float32)
        miou[i][max_check == 1] = (check * (iou_1 + iou_2) / 2)[max_check == 1]

    return miou

def measure_IoU(X1, X2):
    X = X1 + X2

    X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
    X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

    iou = X_inter / X_uni

    check = (X_inter > 0)

    return iou, check

class Evaluation_Function(object):

    def __init__(self, cfg=None):

        self.cfg = cfg
        self.height = 400   # cfg.height, 400
        self.width = 400    # cfg.width, 400

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.X, self.Y = np.meshgrid(np.linspace(0, self.width - 1, self.width),
                                     np.linspace(0, self.height - 1, self.height))
        self.X = torch.tensor(self.X, dtype=torch.float, requires_grad=False).cuda()
        self.Y = torch.tensor(self.Y, dtype=torch.float, requires_grad=False).cuda()

    def measure_miou(self, out, gt):
        ### mask
        output_mask = divided_region_mask(line_pts=out,
                                          size=[self.width, self.height])
        gt_mask = divided_region_mask(line_pts=gt,
                                      size=[self.width, self.height])

        # miou
        precision_miou = measure_IoU_set(ref_mask=gt_mask,
                                         tar_mask=output_mask)
        recall_miou = precision_miou.clone().permute(1, 0).contiguous()

        return precision_miou, recall_miou

    def matching(self, miou):

        out_num, gt_num = miou['p'].shape

        precision = torch.zeros((out_num), dtype=torch.float32).cuda()
        recall = torch.zeros((gt_num), dtype=torch.float32).cuda()

        for i in range(out_num):
            if gt_num == 0:
                break

            max_idx = torch.argmax(miou['p'].view(-1))

            if miou['p'].view(-1)[max_idx] == -1:
                continue

            out_idx = max_idx // gt_num
            gt_idx = max_idx % gt_num

            precision[out_idx] = miou['p'].view(-1)[max_idx]
            miou['p'][out_idx, :] = -1
            miou['p'][:, gt_idx] = -1

        for i in range(gt_num):
            if out_num == 0:
                break

            max_idx = torch.argmax(miou['r'].view(-1))

            if miou['r'].view(-1)[max_idx] == -1:
                continue

            gt_idx = max_idx // out_num
            out_idx = max_idx % out_num

            recall[gt_idx] = miou['r'].view(-1)[max_idx]
            miou['r'][gt_idx, :] = -1
            miou['r'][:, out_idx] = -1

        return precision, recall

    def calculate_AUC(self, miou, metric):

        num = 200
        thresds = np.float32(np.linspace(0, 1, num + 1))
        result = torch.zeros((num + 1), dtype=torch.float32)

        for i in range(thresds.shape[0]):
            thresd = thresds[i]

            correct = 0
            error = 0
            for j in miou[metric[0]]:

                if miou[metric[0]][j].shape[0] != 0:
                    is_correct = (miou[metric[0]][j] > thresd)
                    correct += float(torch.sum(is_correct))
                    error += float(torch.sum(is_correct == 0))

            if (correct + error) != 0:
                result[i] = correct / (correct + error)

        result = result.cpu().numpy()

        AUC = auc(thresds[10:191], result[10:191]) / 0.9

        return AUC, thresds, result

class Evaluation_Semantic_Line(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.eval_func = Evaluation_Function(cfg)
        self.eval_hiou = Evaluation_HIoU(cfg)

    def measure_HIoU(self, pred, gt, mode='test'):
        HIoU = 0

        self.eval_hiou.update_dataset_name(self.cfg.dataset_name)

        gt = gt.cuda()

        result = self.eval_hiou.run(pred, gt)
        if type(result['IOU']) == str:
            pass
        else:
            HIoU += result['IOU']

        return HIoU

    def measure_HIoU_Detector(self, pred, gt, mode='test'):
        HIoU = 0

        self.eval_hiou.update_dataset_name(self.cfg.dataset_name)

        gt = gt.cuda()

        result = self.eval_hiou.forward_Detector(pred, gt)
        if type(result['IOU']) == str:
            pass
        else:
            HIoU += result['IOU']

        #print('HIoU : {}'.format(HIoU))
        return HIoU

    def measure_PRF(self, pred, gt, mode='test'):
        miou = {'a': {},
                'p': {},
                'r': {}}

        match = {'p': {},
                 'r': {}}

        gt = gt.cuda()
        pred_num = pred.shape[0]
        gt_num = gt.shape[0]

        if gt_num == 0:
            match['r'] = torch.zeros(gt_num, dtype=torch.float32).cuda()
        elif pred_num == 0:
            match['p'] = torch.zeros(pred_num, dtype=torch.float32).cuda()
        else:
            miou['p'], miou['r'] = self.eval_func.measure_miou(pred, gt)
            match['p'], match['r'] = self.eval_func.matching(miou)

        return match

    def measure_AUC_PRF(self, match):
        # performance
        auc_p, thresds, precision = self.eval_func.calculate_AUC(miou=match, metric='precision')
        auc_r, _, recall = self.eval_func.calculate_AUC(miou=match, metric='recall')
        try:
            f = 2 * precision * recall / (precision + recall)  # F1-score
        except:
            f = 0

        f[np.isnan(f).nonzero()[0]] = 0
        auc_f = auc(thresds[10:191], f[10:191]) / 0.9

        return auc_p, auc_r, auc_f

    def measure_EA_score_dict(self, dict_results, mode='test'):
        num = len(dict_results['gt'])

        total_precision = np.zeros(99)
        total_recall = np.zeros(99)
        nums_precision = 0
        nums_recall = 0
        for img_name in dict_results['pred'].keys():
            pred = dict_results['pred'][f'{img_name}'][:, [1, 0, 3, 2]]
            pred = to_np(pred)
            gt = np.int32(to_np(dict_results['gt'][f'{img_name}'])[:, [1, 0, 3, 2]]).tolist()

            for i in range(1, 100):
                p, num_p = caculate_precision(pred.tolist(), gt, thresh=i * 0.01)
                r, num_r = caculate_recall(pred.tolist(), gt, thresh=i * 0.01)
                total_precision[i - 1] += p
                total_recall[i - 1] += r
            nums_precision += num_p
            nums_recall += num_r

        total_recall = total_recall / nums_recall
        total_precision = total_precision / nums_precision
        f = 2 * total_recall * total_precision / (total_recall + total_precision)
        f[np.isnan(f).nonzero()[0]] = 0
        # print('Mean P:', total_precision.mean())
        # print('Mean R:', total_recall.mean())
        # print('Mean F:', f.mean())

        mean_p = total_precision.mean()
        mean_r = total_recall.mean()
        mean_f = f.mean()

        return mean_p, mean_r, mean_f
