from functools import partial
import sys
from typing import Dict, List, Optional, Tuple
import torch


from mountaintop.core.ops.common import (
    add_sos_eos, 
    remove_duplicates_and_blank, 
    padding_list, 
    flip_tensor, 
)
from mountaintop.core.ops.mask import (
    make_valid_mask, 
    mask_finished_preds,
    mask_finished_scores, 
    subsequent_mask
)
from mountaintop.core.beam_search import BeamSearchCtc
from mountaintop.layers.base.interface import LayerInterface
from mountaintop.models.base import ModelInterface
from mountaintop.models.asr.utils import EncoderMode

class AsrTransformer(ModelInterface):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        embed: LayerInterface, 
        encoder: LayerInterface,
        decoder: LayerInterface,
        vocab_size: int, 
        ignore_id: int = -1,
        reverse_weight: float = 0.0,
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

        self.reverse_weight = reverse_weight
        
        self._vocab_size = vocab_size
        self._ignore_id = ignore_id
        self._sos = vocab_size - 1
        self._eos = vocab_size - 1

        self._embed = embed
        self._encoder = encoder
        self._decoder = decoder
        
        self._loss_weight = loss_weight
        self._loss_funcs = loss_funcs
        self._ctc = loss_funcs["ctc_loss"]
        if "mwer_loss" in self._loss_funcs:
            self._loss_funcs["mwer_loss"].set_sos_eos(self._sos, self._eos)
        
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
        # Check that batch_size is unified
        assert (feat.size(0) == feat_lengths.size(0) == tgt.size(0) == tgt_lengths.size(0))
        device = feat.device
        
        # 2. Encoder
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
        # 3a. Attention-decoder branch
        if self._loss_weight.get("att_loss_weight", 0.0) > 0.0:
            # loss_att, _ = self._attention_loss(encoder_out, encoder_mask,
            #                                         tgt, tgt_lengths)
            
            ys_hat = add_sos_eos(tgt, tgt_lengths, self._sos, self._eos)
            ys_in_pad = ys_hat[:, :-1]
            ys_out_pad = ys_hat[:, 1:]
            ys_in_lens = tgt_lengths + 1

            # 1. Forward decoder
            decoder_out, r_decoder_out, _ = self._decoder(encoder_out, encoder_mask,
                                                        ys_in_pad, ys_in_lens,
                                                        self.reverse_weight>0.0)
            
            # 2. Compute attention loss
            loss_att = self._loss_funcs["att_loss"](decoder_out, ys_out_pad, ys_in_lens)
            r_loss_att = torch.tensor(0.0).to(device)
            if self.reverse_weight > 0.0:
                #TODO: flipping r_decoder_out
                # r_decoder_out = flip_tensor(r_decoder_out, ys_in_lens, start=0)
                r_ys_out_pad = flip_tensor(ys_out_pad, ys_in_lens-1, start=0)
                r_loss_att = self._loss_funcs["att_loss"](r_decoder_out, r_ys_out_pad, ys_in_lens)
            loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
            
            metrics["att_loss"] = loss_att
        
        # 4b. CTC branch
        if self._loss_weight.get("ctc_loss_weight", 0.0) > 0.0:
            loss_ctc = self._loss_funcs["ctc_loss"](encoder_out, encoder_out_lens, tgt,
                                tgt_lengths)
            metrics["ctc_loss"] = loss_ctc
            
        if self._loss_weight.get("mwer_loss_weight", 0.0) > 0.0:
            ctc_probs = self.ctc_activation(encoder_out).to("cpu")
            beam_size = 10
            beam_decoder = BeamSearchCtc(nbest=beam_size, beam_size=2*beam_size, beam_size_token=2*beam_size)
            beam_hyps = beam_decoder.decode(ctc_probs)
            
            ## reorganize beam results
            beam_decode_results = []
            beam_decode_lengths = []
            for hyps_i in beam_hyps:
                hyps = [hyp.tokens.tolist() for hyp in hyps_i]
                hyps_lens = [len(hyp.tokens.tolist()) for hyp in hyps_i]
                beam_decode_results.append(hyps)
                beam_decode_lengths.append(hyps_lens)
            
            # M-WER loss computation
            forward_fn = partial(self._internal_forward_decoder, use_right_decoder=self.reverse_weight>0.0)
            loss_mwer = self._loss_funcs["mwer_loss"](
                beam_decode_results = beam_decode_results, # [batch_size, beam_size, hyp_len]
                beam_decode_lengths = beam_decode_lengths, # [batch_size, beam_size]
                encoder_out = encoder_out, # [batch_size, feat_len, hidden_size]
                tgt = tgt, 
                tgt_lengths = tgt_lengths,
                forward_fn = forward_fn,
            )
            metrics["mwer_loss"] = loss_mwer
        
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

    def _internal_forward_decoder(
        self, 
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        use_right_decoder: bool = False
    ):
        decoder_out, r_decoder_out, _ = self._decoder(
            encoder_out, encoder_mask, hyps, hyps_lens,
            use_right_decoder)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out
    
    
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
    
    def decode_by_attention(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: str = "offline",
        beam_size: int = 10,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            feat (torch.Tensor): (batch, max_len, feat_dim)
            feat_length (torch.Tensor): (batch, )
            mode (str)
            beam_size (int): beam size for beam search
            chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert feat.size(0) == feat_lengths.size(0)
        assert chunk_size != 0
        device = feat.device
        batch_size = feat.size(0)
        decode_mode = EncoderMode.to_enum(mode)
        
        # 1. Encoder
        encoder_out, encoder_mask = self._internal_forward_encoder(
            feat=feat,
            feat_lengths=feat_lengths, 
            mode=decode_mode,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks, 
        )  # (batch_size, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (batch_size*beam_size, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (batch_size*beam_size, 1, max_len)

        hyps = self._sos * torch.ones([running_size, 1], dtype=torch.long,
                          device=device)  # (batch_size*beam_size, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (batch_size*beam_size, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (batch_size*beam_size, i, i)
            # logp: (batch_size*beam_size, vocab)
            logp, cache = self._decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (batch_size*beam_size, beam_size)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self._eos)
            # 2.3 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp  # (batch_size*beam_size, beam_size), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (batch_size, beam_size*beam_size)
            scores, offset_k_index = scores.topk(k=beam_size)  # (batch_size, beam_size)
            scores = scores.view(-1, 1)  # (batch_size*beam_size, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (batch_size*beam_size*beam_size),regard offset_k_index as (batch_size*beam_size),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (batch_size, beam_size)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (batch_size*beam_size)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (batch_size*beam_size)
            best_hyps_index = torch.div(best_k_index, beam_size, rounding_mode='trunc')
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (batch_size*beam_size, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (batch_size*beam_size, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self._eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_index = torch.argmax(scores, dim=-1).long()
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps
    
    def decode_by_rescore(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: str = "offline",
        beam_size: int = 10,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        # simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        *args, **kwargs
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            feat (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            mode (str): 
            beam_size (int): beam size for beam search
            chunk_size (int): decoding chunk for dynamic chunk trained model
                -1: use full chunk.
                >0: use fixed chunk size as set.
                0: not valid
            num_left_chunks(int): num of chunk of left context
                -1: use all left context.
                >=0: use fixed num of left chunks as set.
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        # only support batch_size=1
        assert feat.size(0) == feat_lengths.size(0)
        assert chunk_size == -1 or chunk_size > 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert self._decoder.is_bidirectional()
        device = feat.device
        decode_mode = EncoderMode.to_enum(mode)
        
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyp_scores, encoder_out = self._internal_ctc_beam(
            feat=feat, 
            feat_lengths=feat_lengths,
            mode=decode_mode,
            beam_size=beam_size,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks
        )

        batch_size = feat.size(0)
        assert len(hyp_scores) == batch_size

        hyps = []
        for hyps_scores_pair in hyp_scores:
            hyps.extend([hyp for hyp, _ in hyps_scores_pair])
        
        hyps_pad = padding_list([torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps], self._ignore_id)
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps], device=device, dtype=torch.long)  # (beam_size,)
        hyps_pad = add_sos_eos(hyps_pad, hyps_lens, self._sos, self._eos)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        hyps_in = hyps_pad[:, :-1]
        
        # encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_out = torch.repeat_interleave(encoder_out, repeats=beam_size, dim=0)
        encoder_mask = torch.ones(size=(batch_size*beam_size, 1, encoder_out.size(1)),
                                  dtype=torch.bool, device=device)
        
        decoder_out, r_decoder_out = self._internal_forward_decoder(
            encoder_out=encoder_out,
            encoder_mask=encoder_mask,
            hyps=hyps_in,
            hyps_lens=hyps_lens,
            use_right_decoder=reverse_weight>0.0
        )
        decoder_out = decoder_out.detach().cpu().numpy()
        r_decoder_out = r_decoder_out.detach().cpu().numpy()

        hyps = []
        best_scores = []
        for i, hyp_score_pair in enumerate(hyp_scores):
            # Only use decoder score for rescoring
            best_score = -float('inf')
            best_index = 0
            for j, hyp_score in enumerate(hyp_score_pair):
                idx = i*beam_size + j
                score = 0.0
                for k, w in enumerate(hyp_score[0]):
                    score += decoder_out[idx][k][w]
                score += decoder_out[idx][len(hyp_score[0])][self._eos]
                # add right to left decoder score
                if reverse_weight > 0:
                    r_score = 0.0
                    for k, w in enumerate(hyp_score[0]):
                        r_score += r_decoder_out[idx][len(hyp_score[0]) - k - 1][w]
                    r_score += r_decoder_out[idx][len(hyp_score[0])][self._eos]
                    score = score * (1 - reverse_weight) + r_score * reverse_weight
                # add ctc score
                score += hyp_score[1] * ctc_weight
                if score > best_score:
                    best_score = score
                    best_index = j
            hyps.append(hyp_score_pair[best_index][0])
            best_scores.append(best_score)
        return hyps, best_scores

    
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

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            True of False
        """
        return self._decoder.is_bidirectional()

    
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

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self._ctc.log_softmax(xs)

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        use_right_decoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call.
            Forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            use_right_decoder: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        assert hyps_lens.size(0) == hyps.size(0)
        num_hyps = hyps.size(0)
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(size=(num_hyps, 1, encoder_out.size(1)),
                                  dtype=torch.bool, device=encoder_out.device)
        decoder_out, r_decoder_out = self._internal_forward_decoder(
            encoder_out=encoder_out,
            encoder_mask=encoder_mask,
            hyps=hyps,
            hyps_lens=hyps_lens,
            use_right_decoder=use_right_decoder
        )
        return decoder_out, r_decoder_out

