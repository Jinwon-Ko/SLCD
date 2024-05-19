import torch
import torch.nn.functional as F


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 0, True, True)
    pred = pred.t()
    correct = pred.eq(target.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sigmoid_focal_loss(inputs, targets, num_lines, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_lines


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def inverse_tanh(x, eps=1e-3):
    x = x.clamp(min=-1, max=1)
    # torch.arctanh(x)
    x1 = (x + 1).clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return 0.5 * torch.log(x1/x2)

def coord_norm(cfg, pts):
    new_pts = pts / (cfg.image_size - 1)
    return new_pts

def coord_unnorm(cfg, pts):
    new_pts = pts * (cfg.image_size - 1)
    return new_pts

def theta_norm(cfg, pts):
    new_pts = pts / cfg.max_theta
    return new_pts

def theta_unnorm(cfg, pts):
    new_pts = pts * cfg.max_theta
    return new_pts

def radius_norm(cfg, pts):
    new_pts = pts / cfg.max_radius
    return new_pts

def radius_unnorm(cfg, pts):
    new_pts = pts * cfg.max_radius
    return new_pts
