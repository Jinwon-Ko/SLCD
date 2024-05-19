import torch
from utils.misc import inverse_sigmoid


def line_denoising(dn_args, num_queries, hidden_dim, dn_enc):
    targets, dn_number, label_noise_ratio, line_noise_scale, dn_theta, dn_radius = dn_args
    dn_number = dn_number * 2  # positive and negative dn queries
    known = [torch.ones_like(v[:, 0][v[:, 0] != -1000], dtype=torch.long) for v in targets]
    batch_size = len(known)
    known_num = [sum(k) for k in known]
    if int(max(known_num)) == 0:
        dn_number = 1
    else:
        if dn_number >= 100:
            dn_number = dn_number // (int(max(known_num) * 2))
        elif dn_number < 1:
            dn_number = 1
    if dn_number == 0:
        dn_number = 1

    # Line Denoising         M: # of GT      P: # of denoising groups (=dn_number)
    lines = torch.cat([v[v[:, 0] != -1000] for v in targets])
    batch_idx = torch.cat([torch.full_like(v[:, 0][v[:, 0] != -1000].long(), i) for i, v in enumerate(targets)])

    known_batch_idx = batch_idx.repeat(2 * dn_number, 1).view(-1)  # [M * 2P]
    known_lines = lines.repeat(2 * dn_number, 1)  # [M * 2P, 2]
    known_line_expand = known_lines.clone()

    positive_idx = torch.tensor(range(len(lines))).long().cuda().unsqueeze(0).repeat(dn_number, 1)  # [P, M]
    positive_idx += (torch.tensor(range(dn_number)) * len(lines) * 2).long().cuda().unsqueeze(1)  # [P, M]
    positive_idx = positive_idx.flatten()  # P * M
    negative_idx = positive_idx + len(lines)  # P * M
    if line_noise_scale > 0:
        known_line_ = known_lines
        # known_line_ = torch.zeros_like(known_lines)         # [M * 2P, 2]
        # known_line_[:, 0] = known_lines[:, 0]
        # known_line_[:, 1] = known_lines[:, 1]

        diff = torch.zeros_like(known_lines)
        diff[:, 0] = dn_theta
        diff[:, 1] = dn_radius

        rand_sign = torch.randint_like(known_lines, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(known_lines)  # 0 <= positive random noise <= 1 : [M * 2P, 2]
        rand_part[negative_idx] += 1.0  # 1 <= negative random noise <= 2 : [M * 2P, 2]
        rand_part *= rand_sign  # Random Noise : [M * 2P, 2]
        known_line_ = known_line_ + torch.mul(rand_part, diff).cuda() * line_noise_scale
        known_line_ = known_line_.clamp(min=0, max=1.0)
        known_line_expand[:, 0] = known_line_[:, 0]
        known_line_expand[:, 1] = known_line_[:, 1]

    # Label Denoising           M: # of GT (total batch)     P: # of denoising groups (=dn_number)
    labels = torch.cat([torch.zeros(len(v[:, 0]), dtype=torch.long)[v[:, 0] != -1000] for v in targets])
    known_labels = labels.repeat(2 * dn_number, 1).view(-1)     # [M * 2P]
    known_labels_expaned = known_labels.clone()                 # [M * 2P]
    known_labels_expaned[negative_idx] = 1                      # 0: Positive / 1: Negative

    m = (1-known_labels_expaned).long().to('cuda')              # [M * 2P]  0: Negative / 1: Positive
    input_label_embed = dn_enc(m)                               # [M * 2P]
    input_line_embed = known_line_expand                        # [M * 2P, 2]

    single_pad = int(max(known_num))
    pad_size = int(single_pad * 2 * dn_number)                  # (maximum M of each batch) * 2P
    padding_line = torch.zeros(pad_size, 2).cuda()              # [max M * 2P, 2]
    padding_label = torch.zeros(pad_size, hidden_dim).cuda()    # [max M * 2P, hidden_dim]

    dn_query_line = padding_line.repeat(batch_size, 1, 1)       # [B, max M * 2P, 2]
    dn_query_feat = padding_label.repeat(batch_size, 1, 1)      # [B, max M * 2P, hidden_dim]

    map_known_indice = torch.tensor([]).to('cuda')
    if len(known_num):
        map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [M]
        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()  # [M * 2P]
    if len(known_batch_idx):
        dn_query_line[(known_batch_idx.long(), map_known_indice)] = input_line_embed    # [B, max M * 2P, 2]
        dn_query_feat[(known_batch_idx.long(), map_known_indice)] = input_label_embed   # [B, max M * 2P]

    # Make attn mask --> a True value indicates that the corresponding position is not allowed to attend
    tgt_size = pad_size + num_queries                           # denoising part + matching part
    dn_attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0   # Initialize with False

    # match query cannot see the denoising groups
    dn_attn_mask[pad_size:, :pad_size] = True

    # denoising groups cannot see each other
    for i in range(dn_number):
        if i == 0:
            dn_attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
        if i == dn_number - 1:
            dn_attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
        else:
            dn_attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            dn_attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

    dn_meta = {
        'pad_size': pad_size,
        'num_dn_group': dn_number,
    }

    return dn_query_feat, dn_query_line, dn_attn_mask, dn_meta


def denoising_post_process(outputs_class, outputs_coord, dn_meta, use_dn, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'cls': output_known_class[-1], 'reg': output_known_coord[-1]}
        if use_dn:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_lines'] = out
    return outputs_class, outputs_coord
