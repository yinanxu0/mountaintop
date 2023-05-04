import math
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def padding_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> padding_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    # batch_size = len(xs)
    # max_len = max([x.size(0) for x in xs])
    # pad = pad_value * torch.ones(batch_size, max_len, dtype=xs[0].dtype, device=xs[0].device)
    # for i in range(batch_size):
    #     pad[i, :xs[i].size(0)] = xs[i]
    
    pad = pad_sequence(xs, True, pad_value)
    return pad


def add_sos_eos(tgt: torch.Tensor, tgt_length: torch.Tensor, sos: int, eos: int) -> torch.Tensor:
    """Add <sos> and <eos> labels.

    Args:
        tgt (torch.Tensor): batch of padded target sequences (batch_size, seqlen)
        tgt_length (torch.Tensor): batch of padded target sequences (batch_size,)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        # ignore_id (int): index of padding

    Returns:
        tgt_hat (torch.Tensor) : (batch_size, seqlen + 2)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        # >>> ignore_id = -1
        >>> tgt
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> tgt_length
        tensor([5, 3, 3], dtype=torch.int32)
        >>> tgt_hat = add_sos_eos(tgt, tgt_length, sos_id , eos_id)
        >>> tgt_hat
        tensor([[10,  1,  2,  3,  4,  5, 11],
                [10,  4,  5,  6, 11, 11, 11],
                [10,  7,  8,  9, 11, 11, 11]])
    """
    batch_size = tgt.size(0)
    max_len = tgt.size(1)
    
    batch_sos = sos*torch.ones((batch_size, 1), device=tgt.device)
    batch_eos = eos*torch.ones((batch_size, 1), device=tgt.device)
    tgt_add_sos = torch.cat([batch_sos, tgt, batch_eos], dim=1) # length + 1
    tgt_length = (tgt_length + 1).unsqueeze(1)
    seq_range = torch.arange(0, max_len+2, dtype=torch.long,
                             device=tgt.device).unsqueeze(0).expand(batch_size, max_len+2)
    valid_tgt_idx = (seq_range < tgt_length).type(torch.long)
    
    tgt_hat = (valid_tgt_idx * tgt_add_sos + (1 - valid_tgt_idx) * eos).to(torch.long)
    return tgt_hat


def flip_tensor(tgt: torch.Tensor, tgt_lengths: torch.Tensor,
                     start: int = 0) -> torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        tgt (tensor): The padded tensor (B, Tokenmax).
        tgt_lengths (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])
    """
    assert tgt.dim() == 2
    assert 0 <= start <= tgt_lengths.min().item(), "0 <= start <= tgt_lengths.min().item()"
    
    tgt_hat = torch.zeros_like(tgt)
    for idx, (y, l) in enumerate(zip(tgt, tgt_lengths)):
        tgt_hat[idx, :start] = y[:start]
        tgt_hat[idx, start:l] = torch.flip(y[start:l], [0])
        tgt_hat[idx, l:] = y[l:]
    return tgt_hat


def th_accuracy(pred_probs: torch.Tensor, true_tgt: torch.Tensor,
                tgt_lengs: torch.Tensor) -> float:
    """Calculate accuracy.

    Args:
        pred_tgt (Tensor): Prediction tensors (B, seqlen, D).
        true_tgt (LongTensor): Target label tensors (B, seqlen).
        tgt_lengs (Tensor): length for target (B,).

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    assert pred_probs.dim() == 3
    assert true_tgt.dim() == 2
    batch_size, seqlen, num_class = pred_probs.size()
    pad_pred = pred_probs.argmax(2) # (B, seqlen)
    ref_idx = torch.arange(0, seqlen, dtype=torch.int64, device=tgt_lengs.device).unsqueeze(0).expand(batch_size, seqlen)
    mask = (ref_idx <= tgt_lengs.unsqueeze(1)).to(torch.int64) # (B, seqlen)
    hit_num = torch.sum(mask * (pad_pred==true_tgt).to(torch.int64))
    total_num = torch.sum(mask)
    return float(hit_num) / float(total_num)


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def log_add(xs: List[int]) -> float:
    """
    Stable log add
    """
    if all(x == -float('inf') for x in xs):
        return -float('inf')
    x_max = max(xs)
    lsp = math.log(sum(math.exp(x - x_max) for x in xs))
    return x_max + lsp
