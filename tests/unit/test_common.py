import pytest
import torch


from mountaintop.core.ops.common import (
    padding_list, 
    add_sos_eos,
    flip_tensor,
    log_add
)

@pytest.mark.parametrize(
    'num', [100, 1000]
)
def test_padding_list(num):
    x = [torch.ones(idx+1) for idx in range(num)]
    y = padding_list(x, 0)
    assert y.size() == (num, num), "FUNCTION padding_list output size is wrong"
    for idx in range(num):
        assert y[idx].sum() == idx+1, "FUNCTION padding_list output size is wrong"



@pytest.mark.parametrize(
    'sos', [-10]
)
@pytest.mark.parametrize(
    'eos', [-11]
)
def test_add_sos_eos(sos, eos):
    tgt = torch.Tensor([[ 1,  2,  3,  4,  5, -1, -1],
                    [ 4,  5,  6, -1, -1, -1, -1],
                    [ 7,  8,  9, -1, -1, -1, -1]])
    tgt_length = torch.Tensor([5, 3, 3])
    
    tgt_hat = add_sos_eos(tgt, tgt_length, sos=sos, eos=eos)
    for idx in range(len(tgt)):
        t = tgt[idx]
        l = tgt_length[idx].to(torch.int32).item()
        h = tgt_hat[idx]
        assert (h[0] - sos).sum() == 0, "FUNCTION add_sos_eos output wrong sos part"
        assert (h[1:l+1] - t[:l]).sum() == 0, "FUNCTION add_sos_eos output wrong body part"
        assert (h[l+1:] - eos).sum() == 0, "FUNCTION add_sos_eos output wrong eos part"
        



@pytest.mark.parametrize(
    'batch_size', [20]
)
@pytest.mark.parametrize(
    'seq_len', [16, 32]
)
@pytest.mark.parametrize(
    'eos', [-11]
)
def test_flip_padding_tensor(batch_size, seq_len, eos):
    import random
    start_idx = 2
    tgt = torch.randint(low=1, high=seq_len, size=(batch_size, seq_len))
    tgt_length = torch.zeros(size=(batch_size,), dtype=torch.long)
    # construct data matrix
    for idx1 in range(batch_size):
        idx2 = random.randint(start_idx, seq_len-2)
        tgt[idx1, idx2:] = eos
        tgt_length[idx1] = idx2
    
    tgt_hat = flip_tensor(tgt, tgt_length, start_idx)
    
    for idx in range(batch_size):
        t = tgt[idx]
        l = tgt_length[idx].to(torch.int32).item()
        h = tgt_hat[idx]
        assert (h[:start_idx] - t[:start_idx]).sum() == 0, "FUNCTION flip_tensor output wrong head part"
        for j in range(start_idx, l):
            assert (h[j] - t[l-1+start_idx-j]).sum() == 0, "FUNCTION flip_tensor output wrong body part"
        assert (h[l:] - t[l:]).sum() == 0, "FUNCTION flip_tensor output wrong foot part"

