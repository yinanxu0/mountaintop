import torch
import random

from mountaintop.models.asr.utils import EncoderMode

def subsequent_mask(
        size: int,
        chunk_size: int = 1,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. Set chunk_size
    and num_left_chunks for the chunk-based attention mask.

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
        
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    assert chunk_size >= 0
    mask = torch.ones(size, size, device=device, dtype=torch.int32)
    mask = torch.tril(mask, out=mask)
    if chunk_size > 1:
        for idx in range(size):
            # right part -> set 1
            start = idx
            end = min((idx // chunk_size + 1) * chunk_size, size)
            mask[idx, start:end] = 1
            
            # left part -> set 0
            if num_left_chunks >= 0:
                start = 0
                end = max((idx // chunk_size - num_left_chunks) * chunk_size, 0)
                mask[idx, start:end] = 0
    return mask


def add_chunk_mask(size: int, masks: torch.Tensor,
                    use_dynamic_chunk: bool = False,
                    chunk_size: int = 16, num_left_chunks: int = -1, 
                    device: torch.device = torch.device("cpu")):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        num_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    ## params: use_dynamic_chunk, chunk_size, num_left_chunks
    if use_dynamic_chunk:
        if chunk_size < 0:
            chunk_size = size
            num_left_chunks = -1
        elif chunk_size == 0:
            ### V1
            # if random.random() < 0.5:
            #     chunk_size = size
            #     num_left_chunks = -1
            # else:
            #     chunk_size = random.randint(1, 25)
            #     max_left_chunks = (size - 1) // chunk_size
            #     num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()
            ### V2
            # if random.random() < 0.5 and num_left_chunks > 0:
            #     chunk_size = random.randint(1, 25)
            #     max_left_chunks = (size - 1) // chunk_size
            #     num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()
            # else:
            #     chunk_size = size
            #     num_left_chunks = -1
            ## V3
            access_prob = torch.rand(size=(1,)).item()
            if access_prob < 0.6:
                chunk_size = torch.randint(low=1, high=25, size=(1,)).item()
                if num_left_chunks < 0:
                    max_left_chunks = max(1, (size - 1) // chunk_size)
                    num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()
            else:
                chunk_size = size
                num_left_chunks = -1
    elif chunk_size <= 0:
        chunk_size = size
        num_left_chunks = -1
    chunk_masks = subsequent_mask(size, chunk_size,
                                  num_left_chunks, device)  # (L, L)
    chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
    chunk_masks = masks & chunk_masks  # (B, L, L)
    ## remove tails in row direction
    masks = masks.transpose(1, 2)
    chunk_masks = masks & chunk_masks  # (B, L, L)
    
    return chunk_masks


def add_chunk_mask_with_mode(
    size: int, 
    masks: torch.Tensor,
    mode: EncoderMode = EncoderMode.Offline,
    chunk_size: int = 16, 
    num_left_chunks: int = -1, 
    device: torch.device = torch.device("cpu")
):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        num_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    if mode == EncoderMode.Offline:
        chunk_size = size
        num_left_chunks = -1
    elif mode == EncoderMode.DynamicChunk:
        access_prob = torch.rand(size=(1,)).item()
        if access_prob < 0.6:
            chunk_size = torch.randint(low=1, high=25, size=(1,)).item()
            if num_left_chunks < 0:
                max_left_chunks = max(1, (size - 1) // chunk_size)
                num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()
        else:
            chunk_size = size
            num_left_chunks = -1
    elif mode == EncoderMode.StaticChunk or mode == EncoderMode.Stream:
        assert chunk_size > 0
        # assert num_left_chunks >= 0
    else:
        raise Exception("encoder mode not valid")
    
    chunk_masks = subsequent_mask(size, chunk_size,
                                  num_left_chunks, device)  # (L, L)
    chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
    chunk_masks = masks & chunk_masks  # (B, L, L)
    ## remove tails in row direction
    masks = masks.transpose(1, 2)
    chunk_masks = masks & chunk_masks  # (B, L, L)
    return chunk_masks


def make_valid_mask(lengths: torch.Tensor, maxlen: int = -1) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_valid_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        
    """
    batch_size = int(lengths.size(0))
    maxlen = maxlen if maxlen > 0 else int(lengths.max().item()) 
    seq_range = torch.arange(0, maxlen, dtype=torch.int64,
                             device=lengths.device).unsqueeze(0).expand(batch_size, maxlen)
    seq_length = lengths.unsqueeze(-1)
    mask = (seq_range < seq_length).type(torch.int32)
    return mask


def make_reverse_valid_mask(lengths: torch.Tensor, maxlen: int = -1) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_reverse_valid_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    return 1 - make_valid_mask(lengths, maxlen)


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    
    return pred.masked_fill_(finished, eos)

