from typing import Callable, List
import torch
import torch.nn.functional as F
import editdistance

from mountaintop.core.ops.common import (
    add_sos_eos, 
    padding_list, 
)


class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        reduction: str = "mean",
        *args, **kwargs,
    ):
        """ Construct CTC module
        Args:
            in_dim: number of encoder projection units
            num_classes: dimension of outputs
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: 
            reduction (string): reduce the CTC loss into a scalar, value should
                be one of "sum", "mean" and "mean_slot", 
                    "sum": means the total loss of theminput batch.
                    "mean": means the total loss divided by the batch size.
                    "mean_slot": means the total loss divided by the num of tokens.
        """
        super().__init__()
        assert in_dim > 0
        assert num_classes > 0
        assert reduction in ["sum", "mean", "mean_slot"]
        self._dropout = torch.nn.Dropout(dropout_rate)
        self._linear = torch.nn.Linear(in_dim, num_classes)
        self._reduction = reduction
        
        if self._reduction == "mean_slot":
            self.loss = torch.nn.CTCLoss(reduction="mean")
        else:
            self.loss = torch.nn.CTCLoss(reduction="sum")

    def forward(self, y: torch.Tensor, y_lengths: torch.Tensor,
                tgt: torch.Tensor, tgt_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            y: batch of padded hidden state sequences (batch_size, seq_len, in_dim)
            y_lengths: batch of lengths of hidden state sequences (batch_size, )
            tgt: batch of padded character id sequence tensor (batch_size, tgt_len)
            tgt_lengths: batch of lengths of character sequence (batch_size, )
        Returns:
            loss: scalar, ctc loss
        """
        batch_size = y.size(0)
        y_hat = self._linear(self._dropout(y)) # (batch_size, seq_len, num_classes)
        y_hat = y_hat.transpose(0, 1) # (seq_len, batch_size, num_classes)
        y_probs = y_hat.log_softmax(2)
        loss = self.loss(y_probs, tgt, y_lengths, tgt_lengths)
        if self._reduction == "mean":
            loss = loss / batch_size
        return loss

    @torch.jit.export
    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            x: torch.Tensor, 3d tensor (batch_size, seq_len, in_dim)
        Returns:
            y: torch.Tensor, log softmax applied 3d tensor (batch_size, seq_len, num_classes)
        """
        return F.log_softmax(self._linear(x), dim=2)

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            x: torch.Tensor, 3d tensor (batch_size, seq_len, in_dim)
        Returns:
            y: torch.Tensor, argmax applied 2d tensor (batch_size, seq_len)
        """
        return torch.argmax(self._linear(x), dim=2)


class LabelSmoothingKlLoss(torch.nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities are taken from the 
    true label prob (1.0) and are divided among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        reduction (string): value should be one of "sum", "mean" and "mean_slot", 
            "sum" means the total loss of theminput batch.
            "mean" means the total loss divided by the batch size.
            "mean_slot" means the total loss divided by the num of tokens.
    """
    def __init__(self,
                 smoothing: float,
                 reduction: str = "mean",
                 *args, **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super().__init__()
        assert reduction in ["sum", "mean", "mean_slot"]
        assert 0.0 <= smoothing <= 1.0
        self._label_prob = 1.0 - smoothing
        self._reduction = reduction
        self._kl_loss = torch.nn.KLDivLoss(reduction="none")

    def forward(self, x: torch.Tensor, target: torch.Tensor, target_length:torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, num_class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, num_class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.ndim == 3
        assert target.ndim == 2
        assert x.shape[:2] == target.shape
        batch_size, max_len, num_classes = x.shape
        x = x.view(-1, num_classes) #(batch*seqlen, num_classes)
        target = target.clone().reshape(-1) # (batch*seqlen,)

        valid_mask = torch.arange(0, max_len, dtype=torch.int32).to(x.device)
        valid_mask = (valid_mask.unsqueeze(0) < target_length.unsqueeze(1)).reshape(-1)
        ignored = ~valid_mask
        
        target = torch.where(ignored, torch.zeros_like(target), target)
        true_dist = torch.nn.functional.one_hot(
            target, num_classes=num_classes
        ).to(x.device)
        
        one_prob = self._label_prob - (1 - self._label_prob) / (num_classes-1)
        zero_prob = (1 - self._label_prob) / (num_classes-1)
        true_dist = true_dist * one_prob + zero_prob
        # Set the value of ignored indexes to 0
        true_dist = torch.where(
            ignored.unsqueeze(1).repeat(1, true_dist.shape[1]),
            torch.zeros_like(true_dist),
            true_dist,
        )
        loss = self._kl_loss(torch.log_softmax(x, dim=1), true_dist)
        if self._reduction == "sum":
            loss_value = loss.sum()
        elif self._reduction == "mean":
            loss_value = loss.sum() / batch_size
        else:
            loss_value = loss.sum() / (valid_mask).sum()
        return loss_value


class LabelSmoothingCeLoss(torch.nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities are taken from the 
    true label prob (1.0) and are divided among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        reduction (string): value should be one of "sum", "mean" and "mean_slot", 
            "sum" means the total loss of theminput batch.
            "mean" means the total loss divided by the batch size.
            "mean_slot" means the total loss divided by the num of tokens.
    """
    def __init__(self,
                 smoothing: float,
                 reduction: str = "mean",
                 *args, **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super().__init__()
        assert reduction in ["sum", "mean", "mean_slot"]
        assert 0.0 <= smoothing <= 1.0
        self._label_prob = 1.0 - smoothing
        self._reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor, target_length:torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, num_class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, num_class)
            target (torch.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.ndim == 3
        assert target.ndim == 2
        assert x.shape[:2] == target.shape
        batch_size, max_len, num_classes = x.shape
        x = x.view(-1, num_classes) #(batch*seqlen, num_classes)
        target = target.clone().reshape(-1) # (batch*seqlen,)

        valid_mask = torch.arange(0, max_len, dtype=torch.int32).to(x.device)
        valid_mask = (valid_mask.unsqueeze(0) < target_length.unsqueeze(1)).reshape(-1)
        ignored = ~valid_mask
        
        target = torch.where(ignored, torch.zeros_like(target), target)
        true_dist = torch.nn.functional.one_hot(
            target.to(torch.int64), num_classes=num_classes
        ).to(x.device)
        
        one_prob = self._label_prob - (1 - self._label_prob) / (num_classes-1)
        zero_prob = (1 - self._label_prob) / (num_classes-1)
        true_dist = true_dist * one_prob + zero_prob
        # Set the value of ignored indexes to 0
        true_dist = torch.where(
            ignored.unsqueeze(1).repeat(1, true_dist.shape[1]),
            torch.zeros_like(true_dist),
            true_dist,
        )
        loss = -1 * (torch.log_softmax(x, dim=1) * true_dist)
        if self._reduction == "sum":
            loss_value = loss.sum()
        elif self._reduction == "mean":
            loss_value = loss.sum() / batch_size
        else:
            loss_value = loss.sum() / (valid_mask).sum()
        return loss_value


class MwerLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs) -> None:
        super().__init__()
        self._reduction = reduction
        self._sos = None
        self._eos = None
    
    def set_sos_eos(self, sos, eos):
        self._sos = sos
        self._eos = eos
    
    def forward(
        self, 
        beam_decode_results: List[List],
        beam_decode_lengths: List[List],
        encoder_out: torch.Tensor,
        tgt: torch.Tensor, 
        tgt_lengths: torch.Tensor,
        forward_fn: Callable,
    ):
        batch_size = len(beam_decode_results)
        beam_size = len(beam_decode_results[0])
        
        assert batch_size == tgt.shape[0] == tgt_lengths.shape[0]
        loss = torch.tensor(0.0, device=encoder_out.device)
        
        for idx in range(batch_size):
            beam_decode_result = beam_decode_results[idx]
            beam_decode_length = beam_decode_lengths[idx]
            beam_size = len(beam_decode_result)
            curr_tgt = tgt[idx][:tgt_lengths[idx]].cpu().detach().numpy().tolist()
            
            # 获得 P(y_i | x)，注意，这个必须要梯度回传
            prob_hat = self._get_prob(
                beam_decode_result,
                beam_decode_length,
                encoder_out[idx].unsqueeze(0), 
                forward_fn
            )
            for j in range(beam_size):
                # 计算 W(y_i, y^*)，beam [batch_size, seq_len, ]
                wer = editdistance.distance(beam_decode_result[j], curr_tgt)
                loss += prob_hat[j] * wer
        if self._reduction == "mean":
            loss = loss / batch_size
        return loss

    def _get_prob(
        self, 
        hyps: List, 
        hyps_lens: List,
        encoder_out: torch.Tensor, 
        forward_fn: Callable,
    ):
        device = encoder_out.device
        hyps_pad = padding_list([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], -1)
        hyps_lens = torch.tensor(hyps_lens, device=device, dtype=torch.long)  # (beam_size,)
        hyps_pad = add_sos_eos(hyps_pad, hyps_lens, self._sos, self._eos)
        hyps_in = hyps_pad[:, :-1]
        hyps_out = hyps_pad[:, 1:]
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        
        beam_size = hyps_pad.size(0)
        max_len = torch.max(hyps_lens)
        
        encoder_out = encoder_out.expand(beam_size, -1, -1)
        encoder_mask = torch.ones(
            size=(beam_size, 1, encoder_out.size(1)),
            dtype=torch.bool,
            device=device
        )
        
        decoder_out, r_decoder_out = forward_fn(
            encoder_out,
            encoder_mask,
            hyps_in,
            hyps_lens
        )
 
        # 取出beam路径上的概率P
        # decoder_out.shape: [beam_size, seq_len, vocab_size]
        prob_raw = torch.gather(decoder_out, dim=2, index=hyps_out.unsqueeze(-1)).squeeze(-1)
        # 将padding位置的log softmax值设为常量，不参与梯度反传
        valid_mask = torch.arange(0, max_len, dtype=torch.int32).to(device)
        valid_mask = valid_mask.unsqueeze(0) > hyps_lens.unsqueeze(1)
        prob_valid = torch.where(valid_mask, torch.zeros_like(prob_raw), prob_raw)
        # 这里已经使用了log_softmax，直接相加
        prob = torch.sum(prob_valid, dim=-1)
        prob_hat = torch.nn.functional.softmax(prob, dim=-1)
        # # 注意这里减了最大值，相当于分子分母同时exp{-max{p_i}}，防止变成零
        # prob = torch.exp(prob - torch.max(prob))
        # prob_sum = torch.sum(prob, dim=-1)
        # prob_hat = prob / prob_sum
        return prob_hat



class FastMwerLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs) -> None:
        super().__init__()
        self._reduction = reduction
        self._sos = None
        self._eos = None
    
    def set_sos_eos(self, sos, eos):
        self._sos = sos
        self._eos = eos
    
    def forward(
        self, 
        beam_decode_results: List[List],
        beam_decode_lengths: List[List],
        encoder_out: torch.Tensor,
        tgt: torch.Tensor, 
        tgt_lengths: torch.Tensor,
        forward_fn: Callable,
    ):
        batch_size = len(beam_decode_results)
        beam_size = len(beam_decode_results[0])
        
        assert batch_size == tgt.shape[0] == tgt_lengths.shape[0]
        
        ## compute wer
        wers = []
        for i in range(batch_size):
            curr_tgt = tgt[i][:tgt_lengths[i]].cpu().detach().numpy().tolist()
            wer = [editdistance.distance(result, curr_tgt) for result in beam_decode_results[i]]
            wers.append(wer)
        
        probs = torch.zeros(
            size=(batch_size, beam_size), 
            dtype=encoder_out.dtype,
            device=encoder_out.device
        )
        for j in range(beam_size):
            batch_data = [beam_decode_results[i][j] for i in range(batch_size)]
            batch_length = [beam_decode_lengths[i][j] for i in range(batch_size)]
            
            # 获得 P(y_i | x)，注意，这个必须要梯度回传
            probs[:, j] = self._fast_get_prob(
                batch_data,
                batch_length,
                encoder_out, 
                forward_fn
            )
        
        loss = torch.tensor(0.0, device=encoder_out.device)
        # 计算 W(y_i, y^*)，beam [batch_size, seq_len, ]
        for i in range(batch_size):
            prob_norm = torch.nn.functional.softmax(probs[i], dim=-1)
            for j in range(beam_size):
                loss += prob_norm[j] * wers[i][j]

        if self._reduction == "mean":
            loss = loss / batch_size
        return loss

    def _fast_get_prob(
        self,
        hyps: List, 
        hyps_lens: List,
        encoder_out: torch.Tensor, 
        forward_fn: Callable,
    ):
        device = encoder_out.device
        hyps_pad = padding_list([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], -1)
        hyps_lens = torch.tensor(hyps_lens, device=device, dtype=torch.long)  # (beam_size,)
        hyps_pad = add_sos_eos(hyps_pad, hyps_lens, self._sos, self._eos)
        hyps_in = hyps_pad[:, :-1]
        hyps_out = hyps_pad[:, 1:]
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        
        batch_size = hyps_pad.size(0)
        max_len = torch.max(hyps_lens)
        
        encoder_mask = torch.ones(
            size=(batch_size, 1, encoder_out.size(1)),
            dtype=torch.bool,
            device=device
        )
        
        decoder_out, r_decoder_out = forward_fn(
            encoder_out,
            encoder_mask,
            hyps_in,
            hyps_lens
        )
        prob_raw = torch.gather(decoder_out, dim=2, index=hyps_out.unsqueeze(-1)).squeeze(-1)
        # 将padding位置的log softmax值设为常量，不参与梯度反传
        valid_mask = torch.arange(0, max_len, dtype=torch.int32).to(device)
        valid_mask = valid_mask.unsqueeze(0) > hyps_lens.unsqueeze(1)
        prob_valid = torch.where(valid_mask, torch.zeros_like(prob_raw), prob_raw)
        # 这里已经使用了log_softmax，直接相加
        prob = torch.sum(prob_valid, dim=-1)
        return prob


class MaeLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean", *args, **kwargs):
        super().__init__()
        assert reduction in ["sum", "mean", "mean_slot"]
        self._reduction = reduction
        self.loss = torch.nn.L1Loss(reduction='sum')

    def forward(self, token_length: torch.Tensor, pre_token_length: torch.Tensor):
        loss = self.loss(token_length, pre_token_length)
        if self._reduction == "mean":
            batch_size = token_length.size(0)
            loss_value = loss / batch_size
        elif self._reduction == "mean_slot":
            num_tokens = token_length.sum().type(torch.float32)
            loss_value = loss / num_tokens
        else:
            loss_value = loss
        return loss_value
