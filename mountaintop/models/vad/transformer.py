from functools import partial
import sys
from typing import Dict, List, Optional, Tuple
import torch


from mountaintop.core.ops.common import remove_duplicates_and_blank
from mountaintop.core.ops.mask import make_valid_mask
from mountaintop.core.beam_search import BeamSearchCtc
from mountaintop.layers.base.interface import LayerInterface
from mountaintop.models.base import ModelInterface
from mountaintop.models.asr.utils import EncoderMode


class VadTransformer(ModelInterface):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        embed: LayerInterface, 
        encoder: LayerInterface,
        vocab_size: int, 
        loss_funcs: Dict[str, torch.nn.Module] = {},
        loss_weight: Dict[str, float] = {},
        train_mode: str = "offline",
        **kwargs,
    ):
        super().__init__()
        for k, v in loss_weight.items():
            assert 0.0 <= v <= 1.0, f"{k} should be positive"
        # assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert vocab_size > 0
        # note that eos is the same as sos (equivalent ID)
        assert train_mode in ["offline", "dynamicchunk", "dynamic_chunk", "staticchunk", "static_chunk"]

        self._vocab_size = vocab_size
        self._sos = vocab_size - 1
        self._eos = vocab_size - 1

        self._embed = embed
        self._encoder = encoder
        
        self._loss_weight = loss_weight
        self._loss_funcs = loss_funcs
        # self._ctc = loss_funcs["ctc_loss"]
        self._projection = torch.nn.Linear(self._encoder.dim, self._vocab_size)

        self._train_mode = EncoderMode.to_enum(train_mode) 
        

    '''
    To be used in parallel training, must use default forward function. Other
    funtions could not be found under DistributedDataParallel Model.
    '''
    @torch.jit.ignore
    def forward(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        tgt: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            feat: (Batch, Length, ...)
            feat_lengths: (Batch, )
            tgt: (Batch, Length)
            tgt_lengths: (Batch,)
        """
        assert feat_lengths.dim() == 1, feat_lengths.shape
        assert tgt_lengths.dim() == 1, tgt_lengths.shape
        # Check the batch_size
        assert (feat.size(0) == feat_lengths.size(0) == tgt.size(0) == tgt_lengths.size(0))
        device = feat.device
        
        # 1, Encoder
        encoder_out, encoder_mask = self._internal_forward_encoder(
            feat=feat,
            feat_lengths=feat_lengths,
            mode=self._train_mode,
            # chunk_size=0,
            # num_left_chunks=-1,
            # simulate_streaming=False
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        metrics = {}

        # 2.1, CTC branch
        if self._loss_weight.get("ctc_loss_weight", 0.0) > 0.0:
            loss_ctc = self._loss_funcs["ctc_loss"](encoder_out, encoder_out_lens, tgt,
                                tgt_lengths)
            metrics["ctc_loss"] = loss_ctc
        
        # 2.2, CE branch
        if self._loss_weight.get("ce_loss_weight", 0.0) > 0.0:
            sampling_target = tgt[:, ::self.subsampling_rate()]
            memory_actual_size1 = min(encoder_out.shape[1], sampling_target.shape[1])
            prob_hat = self._projection(encoder_out[:, :memory_actual_size1])
            loss_ce = self._loss_funcs["ce_loss"](
                prob_hat, 
                sampling_target[:, :memory_actual_size1], 
                encoder_out_lens
            )
            metrics["ce_loss"] = loss_ce

        loss_total = torch.tensor(0.0, device=device)
        for key in metrics.keys():
            loss_total += metrics[key] * self._loss_weight[f"{key}_weight"]
        metrics["loss"] = loss_total
        predictions = {}
        return predictions, loss_total, metrics


    ##############################
    ##### internal functions #####
    ##############################
    def _internal_forward_encoder(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: EncoderMode = EncoderMode.DynamicChunk, 
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        # simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check that batch_size is unified
        assert (feat.size(0) == feat_lengths.size(0))
        masks = make_valid_mask(feat_lengths, maxlen=feat.size(1)).unsqueeze(1)  # (batch_size, 1, T)
        xs, masks, pos_emb = self._embed(feat, masks)
        
        if mode != EncoderMode.Stream:
            ## non-stream mode, one-time encoding 
            encoder_out, encoder_mask = self._encoder(
                feat=xs, 
                masks=masks, 
                pos_emb=pos_emb,
                mode=mode,
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks
            )
        elif mode == EncoderMode.Stream:
            ## stream mode
            assert chunk_size > 0
            encoder_out, encoder_mask = self._encoder.forward_stream(
                feat=xs, 
                pos_emb=pos_emb,
                chunk_size=chunk_size, 
                num_left_chunks=num_left_chunks
            )  # (batch_size, seqlen, encoder_dim)
        else:
            raise Exception("wrong Encoder Mode")

        return encoder_out, encoder_mask
        
    def _internal_ctc_beam(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: EncoderMode = EncoderMode.Offline,
        beam_size: int = 10,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        # simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            feat (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        # only support batch_size=1
        assert feat.size(0) == feat_lengths.size(0)
        assert chunk_size != 0
        
        # 1. Encoder forward and get CTC score
        encoder_out, _ = self._internal_forward_encoder(
            feat=feat, 
            feat_lengths=feat_lengths, 
            mode=mode,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks,
        )  # (batch_size, maxlen, encoder_dim)
        
        ctc_probs = self.ctc_activation(encoder_out).to("cpu")  # (batch_size, maxlen, vocab_size)
        
        beam_decoder = BeamSearchCtc(nbest=beam_size, beam_size=2*beam_size, beam_size_token=2*beam_size)
        hyps = beam_decoder.decode(ctc_probs)
        hyps_scores_pairs = []
        for hyp in hyps:
            hyps_scores_pair = [(tuple(h.tokens.tolist()), h.score) for h in hyp]
            hyps_scores_pairs.append(hyps_scores_pair)
        
        return hyps_scores_pairs, encoder_out


    ##############################
    ##### decoding functions #####
    ##############################
    def decode_by_ctc_greedy(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: str = "offline",
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        # simulate_streaming: bool = False,
        *args, **kwargs
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            feat (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert feat.size(0) == feat_lengths.size(0)
        assert chunk_size != 0
        batch_size = feat.size(0)
        # maxlen = feat.size(1)
        decode_mode = EncoderMode.to_enum(mode)
        
        encoder_out, encoder_mask = self._internal_forward_encoder(
            feat=feat,
            feat_lengths=feat_lengths, 
            mode=decode_mode,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks, 
        )  # (batch_size, maxlen, encoder_dim)
        maxlen = encoder_out.size(1) # with subsampling, encoder_out.size(1) < feat.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc_activation(encoder_out)  # (batch_size, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (batch_size, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (batch_size, maxlen)
        mask = make_valid_mask(encoder_out_lens, maxlen=maxlen)  # (batch_size, maxlen)
        topk_index = topk_index*mask + (1-mask)*self._eos
        hyps = [remove_duplicates_and_blank(hyp.tolist()) for hyp in topk_index]
        return hyps
    
    def decode_by_ctc_beam(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: str = "offline",
        beam_size: int = 10,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        # simulate_streaming: bool = False,
        *args, **kwargs
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            feat (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        decode_mode = EncoderMode.to_enum(mode)
        hyps_scores_pairs, _ = self._internal_ctc_beam(
            feat=feat, 
            feat_lengths=feat_lengths,
            mode=decode_mode,
            beam_size=beam_size, 
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks,
        )
        hyps = [hyps_scores_pair[0][0] for hyps_scores_pair in hyps_scores_pairs]
        return hyps

    
    ####################
    ##### property #####
    ####################
    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self._embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self._embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self._sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self._eos


    ###################
    ##### methods #####
    ###################
    @torch.jit.export
    def forward_encoder_chunk(
        self,
        feat: torch.Tensor,
        offset: int,
        required_cache_size: int,
        attn_cache: Optional[List[torch.Tensor]] = None,
        cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """ Export interface for c++ call, give input chunk feat, and return
            output from time 0 to current chunk.

        Args:
            feat (torch.Tensor): chunk input
            subsampling_cache (Optional[torch.Tensor]): subsampling cache
            attn_cache (Optional[List[torch.Tensor]]):
                transformer/conformer encoder layers output cache
            cnn_cache (Optional[List[torch.Tensor]]): conformer
                cnn cache

        Returns:
            torch.Tensor: output, it ranges from time 0 to current chunk.
            List[torch.Tensor]: attention cache
            List[torch.Tensor]: conformer cnn cache

        """
        feat_lengths = torch.tensor(feat.size()[:2])
        masks = make_valid_mask(feat_lengths, maxlen=feat.size(1)).unsqueeze(1)
        feat, masks, pos_emb = self._embed(feat, masks, offset=offset)
        
        return self._encoder.forward_one_chunk(
            feat, 
            pos_emb=pos_emb,
            chunk_size=feat.size(1),
            num_left_chunks=required_cache_size,
            attn_cache=attn_cache,
            cnn_cache=cnn_cache
        )

    # @torch.jit.export
    # def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
    #     """ Export interface for c++ call, apply linear transform and log
    #         softmax before ctc
    #     Args:
    #         xs (torch.Tensor): encoder output

    #     Returns:
    #         torch.Tensor: activation before ctc

    #     """
    #     return self._ctc.log_softmax(xs)
