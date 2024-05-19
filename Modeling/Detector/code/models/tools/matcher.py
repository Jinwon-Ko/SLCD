import torch
import torch.nn as nn

from scipy.optimize import linear_sum_assignment
from utils.util import *

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, args):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_line: This is the relative weight of the L1 error of the line coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the line in the matching cost
        """
        super().__init__()
        self.cfg = args

        self.cost_class = args.set_cost_class
        self.cost_line = args.set_cost_line
        self.focal_alpha = args.focal_alpha

        assert self.cost_class != 0 or self.cost_line != 0, "all costs cant be 0"

    @torch.no_grad()
    def matching_for_line_detection(self, outputs, targets):
        bs, num_queries = outputs['params_0'].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_line = outputs['params_0'].clone().detach().flatten(0, 1)         # [batch_size * num_queries, 2]
        out_prob = outputs['cls'].clone().detach().flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and lines
        tgt_line = torch.cat([v[v[:, 0] != -1000] for v in targets['params']])
        tgt_ids = torch.cat([torch.zeros(len(v[:, 0]), dtype=torch.long)[v[:, 0] != -1000] for v in targets['params']])

        # Compute the L2 cost between lines
        out_line[..., 0] = out_line[..., 0] / (self.cfg.max_theta / 2)     # More sensitive on angle displacement
        tgt_line[..., 0] = tgt_line[..., 0] / (self.cfg.max_theta / 2)     # More sensitive on angle displacement
        out_line[..., 1] = out_line[..., 1] / self.cfg.max_radius
        tgt_line[..., 1] = tgt_line[..., 1] / self.cfg.max_radius
        cost_reg = torch.cdist(out_line, tgt_line, p=2)

        # Compute the classification cost.
        cost_cls = (1 - out_prob)[:, tgt_ids]

        # Final cost matrix
        C = self.cost_line * cost_reg + self.cost_class * cost_cls
        C = C.view(bs, num_queries, -1).cpu()

        # sizes = [len(v["lines"]) for v in targets]
        sizes = [len(torch.nonzero(v[:, 0] != -1000)) for v in targets['params']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(args)
