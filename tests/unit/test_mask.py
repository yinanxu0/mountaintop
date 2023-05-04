import pytest
import torch


from mountaintop.core.ops.mask import (
    subsequent_mask, add_chunk_mask, 
    make_valid_mask, make_reverse_valid_mask, 
    mask_finished_scores, mask_finished_preds
)


@pytest.mark.parametrize(
    'seq_len', [32, 64]
)
@pytest.mark.parametrize(
    # 'chunk_size', [1, 2, 3]
    'chunk_size', [2]
)
@pytest.mark.parametrize(
    # 'num_left_chunks', [1, 2, 3]
    'num_left_chunks', [1]
)
def test_subsequent_mask(seq_len, chunk_size, num_left_chunks):
    mask = subsequent_mask(seq_len)
    assert mask.size() == (seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i][j] == 1, "mask at ({}, {}) should be 1".format(i, j)
            else:
                assert mask[i][j] == 0, "mask at ({}, {}) should be 0".format(i, j)

    mask = subsequent_mask(seq_len, chunk_size, num_left_chunks)
    assert mask.size() == (seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            chunk_idx = i // chunk_size
            left_idx = (chunk_idx-num_left_chunks)*chunk_size
            right_idx = (chunk_idx+1)*chunk_size
            if left_idx <= j and j < right_idx:
                assert mask[i][j] == 1, "mask at ({}, {}) should be 1".format(i, j)
            else:
                assert mask[i][j] == 0, "mask at ({}, {}) should be 0".format(i, j)


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'seq_len', [32, 64]
)
@pytest.mark.parametrize(
    # 'chunk_size', [1, 2, 3]
    'chunk_size', [2]
)
@pytest.mark.parametrize(
    # 'num_left_chunks', [1, 2, 3]
    'num_left_chunks', [1]
)
def test_add_chunk_mask_positive(batch_size, seq_len, chunk_size, num_left_chunks):
    lengths = torch.randint(low=1, high=seq_len, size=(batch_size,))
    masks = make_valid_mask(lengths, seq_len).unsqueeze(1)
    chunk_mask = add_chunk_mask(seq_len, masks, False, chunk_size, num_left_chunks)
    
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(seq_len):
                if j < lengths[i] and k < lengths[i]:
                    chunk_idx = j // chunk_size
                    left_idx = (chunk_idx-num_left_chunks)*chunk_size
                    right_idx = (chunk_idx+1)*chunk_size
                    if left_idx <= k and k < right_idx:
                        assert chunk_mask[i][j][k] == 1, "chunk mask at ({}, {}, {}) should be 1".format(i, j, k)
                    else:
                        assert chunk_mask[i][j][k] == 0, "chunk mask at ({}, {}, {}) should be 0".format(i, j, k)
                else:
                    assert chunk_mask[i][j][k] == 0, "chunk mask at ({}, {}, {}) should be 0".format(i, j, k)


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'seq_len', [32, 64]
)
@pytest.mark.parametrize(
    # 'chunk_size', [1, 2, 3]
    'chunk_size', [2]
)
@pytest.mark.parametrize(
    # 'num_left_chunks', [1, 2, 3]
    'num_left_chunks', [1]
)
def test_add_chunk_mask_negative(batch_size, seq_len, chunk_size, num_left_chunks):
    chunk_size = -1
    num_left_chunks = -1
    lengths = torch.randint(low=1, high=seq_len, size=(batch_size,))
    masks = make_valid_mask(lengths, seq_len).unsqueeze(1)
    chunk_mask = add_chunk_mask(seq_len, masks, False, chunk_size, num_left_chunks)
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(seq_len):
                if j < lengths[i] and k < lengths[i]:
                    assert chunk_mask[i][j][k] == 1, "chunk mask at ({}, {}, {}) should be 1".format(i, j, k)
                else:
                    assert chunk_mask[i][j][k] == 0, "chunk mask at ({}, {}, {}) should be 0".format(i, j, k)


@pytest.mark.parametrize(
    'batch_size', [20, 30]
)
@pytest.mark.parametrize(
    'seq_len', [32, 64]
)
def test_make_valid_mask(batch_size, seq_len):
    lengths = torch.randint(low=1, high=seq_len, size=(batch_size,))
    mask1 = make_valid_mask(lengths, seq_len)
    mask2 = make_reverse_valid_mask(lengths, seq_len)
    for i in range(batch_size):
        for j in range(seq_len):
            if j < lengths[i]:
                assert mask1[i][j] == 1, "mask at ({}, {}) should be 1".format(i, j)
                assert mask2[i][j] == 0, "mask at ({}, {}) should be 0".format(i, j)
            else:
                assert mask1[i][j] == 0, "mask at ({}, {}) should be 0".format(i, j)
                assert mask2[i][j] == 1, "mask at ({}, {}) should be 0".format(i, j)

