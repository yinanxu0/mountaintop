from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union
)

import torch
from typeguard import check_argument_types


from mountaintop import __version__
from mountaintop.runx.logx import loggerx
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
from mountaintop.core.beam_search import BeamSearchSequence

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class Paraformer(ModelInterface):
    """
    Author: Speech Lab, Alibaba Group, China
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        embed: LayerInterface, 
        encoder: LayerInterface,
        decoder: LayerInterface,
        predictor: LayerInterface,
        vocab_size: int,
        predictor_bias: int = 1, ## TODO: predictor_offset is better?
        sampling_ratio: float = 0.2,
        ignore_id: int = -1,
        reverse_weight: float = 0.0,
        loss_funcs: Dict[str, torch.nn.Module] = {},
        loss_weight: Dict[str, float] = {},
        *args, **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        for k, v in loss_weight.items():
            assert 0.0 <= v <= 1.0, f"{k} should be positive"
        assert vocab_size > 0
        
        super().__init__()
        
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
        # if "mwer_loss" in self._loss_funcs:
        #     self._loss_funcs["mwer_loss"].set_sos_eos(self._sos, self._eos)
        
        self._model_version = str(__version__)
        self._predictor = predictor
        
        self.predictor_bias = predictor_bias
        self.sampling_ratio = sampling_ratio


    @torch.jit.ignore
    def forward(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        tgt: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
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
        assert (feat.size(0) == feat_lengths.size(0) == tgt.size(0) == tgt_lengths.size(0)), \
                (feat.shape, feat_lengths.shape, tgt.shape, tgt_lengths.shape)
        batch_size = feat.shape[0]
        device = feat.device
        
        ######## mountaintop part ########
        ##################################
        metrics = dict()
        # 1, Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            feat=feat,
            feat_lengths=feat_lengths,
            chunk_size=0,
            num_left_chunks=-1,
            simulate_streaming=False
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        
        # 2, CTC Loss
        if self._loss_weight.get("ctc_loss_weight", 0.0) > 0.0:
            loss_ctc = self._ctc(encoder_out, encoder_out_lens, tgt,
                                tgt_lengths)
            metrics["ctc_loss"] = loss_ctc
        
        # 3, Predictor
        if self.predictor_bias == 1:
            ys_hat = add_sos_eos(tgt, tgt_lengths, self._sos, self._eos)
            # ys_in_pad = ys_hat[:, :-1]
            ys_out_pad = ys_hat[:, 1:]
            ys_lens = tgt_lengths + 1
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self._predictor(
            encoder_out, ys_lens, mask=encoder_mask
        )
        if self._loss_weight.get("mae_loss_weight", 0.0) > 0.0:
            loss_mae = self._loss_funcs["mae_loss"](ys_lens.type_as(pre_token_length), pre_token_length)
            metrics["mae_loss"] = loss_mae
    
        # 4, Attention Loss
        if self._loss_weight.get("att_loss_weight", 0.0) > 0.0:
            # 4.1, Sampler
            decoder_out_1st = None
            if self.sampling_ratio > 0.0:
                # if self.step_cur < 2:
                #     loggerx.info("enable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
                sematic_embeds, decoder_out_1st = self._sampler(encoder_out, encoder_mask, ys_out_pad, ys_lens, pre_acoustic_embeds)
            else:
                # if self.step_cur < 2:
                #     loggerx.info("disable sampler in paraformer, sampling_ratio: {}".format(self.sampling_ratio))
                sematic_embeds = pre_acoustic_embeds

            # 4.2, Forward decoder
            decoder_out, r_decoder_out, _ = self._decoder.forward_by_embedding(
                encoder_out, encoder_mask, sematic_embeds, ys_lens
            )
            if decoder_out_1st is None:
                decoder_out_1st = decoder_out
                
            # 4.3, Compute attention loss
            loss_att = self._loss_funcs["att_loss"](decoder_out, ys_out_pad, ys_lens)
            r_loss_att = torch.tensor(0.0).to(device)
            if self.reverse_weight > 0.0:
                #TODO: flipping r_decoder_out
                # r_decoder_out = flip_tensor(r_decoder_out, ys_lens, start=0)
                r_ys_out_pad = flip_tensor(ys_out_pad, ys_lens-1, start=0)
                r_loss_att = self._loss_funcs["att_loss"](r_decoder_out, r_ys_out_pad, ys_lens)
            loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
            metrics["att_loss"] = loss_att
            
        loss_total = torch.tensor(0.0, device=device)
        for key in metrics.keys():
            loss_total += metrics[key] * self._loss_weight[f"{key}_weight"]
        metrics["loss"] = loss_total
        predictions = {}
        return predictions, loss_total, metrics
        

    ## own parts
    def _forward_encoder(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check that batch_size is unified
        assert (feat.size(0) == feat_lengths.size(0))
        masks = make_valid_mask(feat_lengths, maxlen=feat.size(1)).unsqueeze(1)  # (batch_size, 1, T)
        
        xs, masks, pos_emb = self._embed(feat, masks)
        # 1. Encoder
        if simulate_streaming and chunk_size > 0:
            encoder_out, encoder_mask = self._encoder.forward_stream(
                xs, pos_emb, chunk_size=chunk_size, num_left_chunks=num_left_chunks
            )  # (batch_size, seqlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self._encoder(
                xs, masks, pos_emb,
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks
            )  # (batch_size, seqlen, encoder_dim)
        return encoder_out, encoder_mask

    def _sampler(
        self, encoder_out, encoder_mask, 
        ys_pad, ys_lens, pre_acoustic_embeds
    ):
        device = pre_acoustic_embeds.device
        seqlen = ys_pad.size(1)
        tgt_mask = (make_valid_mask(ys_lens, maxlen=seqlen).unsqueeze(2)).to(ys_pad.device)

        # tgt_mask = make_valid_mask(ys_lens, maxlen=ys_lens.max())[:, :, None].to(ys_pad.device)
        ys_pad = ys_pad * tgt_mask[:, :, 0]
        ys_pad_embed, _ = self._decoder.embed(ys_pad)
        
        batch_size = ys_pad.size(0)
        
        with torch.no_grad():
            decoder_out, _, _ = self._decoder.forward_by_embedding(
                encoder_out, encoder_mask, pre_acoustic_embeds, ys_lens
            )
            # decoder_out, _ = decoder_outs[0], decoder_outs[1]
            prediction_tokens = decoder_out.argmax(-1)
            
            ### need modification
            nonpad_positions = ys_pad.ne(self._ignore_id)
            seq_lens = (nonpad_positions).sum(1)
            ###
            
            same_num = ((prediction_tokens == ys_pad) & nonpad_positions).sum(1) # [batch_size, ]
            input_mask = torch.ones_like(nonpad_positions, device=device)
            # bsz, seq_len = ys_pad.size()
            for example_id in range(batch_size):
                target_num = (((seq_lens[example_id] - same_num[example_id]).float()) * self.sampling_ratio).long()
                if target_num > 0:
                    sample_index = torch.randperm(seq_lens[example_id], device=device)[:target_num]
                    input_mask[example_id].scatter_(
                        dim=0, 
                        index=sample_index, 
                        value=0
                    )
            input_mask = input_mask.eq(1)
            input_mask = input_mask.masked_fill(~nonpad_positions, False)
            # input_mask_expand_dim = input_mask.unsqueeze(2).to(pre_acoustic_embeds.device)
            input_mask_expand_dim = input_mask.unsqueeze(2)

        sematic_embeds = pre_acoustic_embeds.masked_fill(
                ~input_mask_expand_dim, 0
            ) + ys_pad_embed.masked_fill(
                input_mask_expand_dim, 0
            )
        return sematic_embeds * tgt_mask, decoder_out * tgt_mask

    ##############################
    ##### internal functions #####
    ##############################
    def _internal_forward_encoder(
        self,
        feat: torch.Tensor,
        feat_lengths: torch.Tensor,
        mode: EncoderMode = EncoderMode.Offline,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check that batch_size is unified
        assert (feat.size(0) == feat_lengths.size(0))
        assert mode.is_valid(), f"Wrong Encoder Mode: {mode}"
        masks = make_valid_mask(feat_lengths, maxlen=feat.size(1)).unsqueeze(1)  # (batch_size, 1, T)
        xs, masks, pos_emb = self._embed(feat, masks)
        
        if mode == EncoderMode.Stream:
            ## stream mode
            assert chunk_size > 0
            encoder_out, encoder_mask = self._encoder.forward_stream(
                feat=xs, 
                pos_emb=pos_emb,
                chunk_size=chunk_size, 
                num_left_chunks=num_left_chunks
            )  # (batch_size, seqlen, encoder_dim)
        else:
            ## non-stream mode, one-time encoding 
            encoder_out, encoder_mask = self._encoder(
                feat=xs, 
                masks=masks, 
                pos_emb=pos_emb,
                mode=mode,
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks
            )

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
    ) -> List[List[int]]:
        assert feat.size(0) == feat_lengths.size(0)
        assert chunk_size != 0
        batch_size = feat.size(0)
        # maxlen = feat.size(1)
        decode_mode = EncoderMode.to_enum(mode)
        assert decode_mode.is_valid()
        
        encoder_out, encoder_mask = self._internal_forward_encoder(
            feat=feat,
            feat_lengths=feat_lengths, 
            mode=decode_mode,
            chunk_size=chunk_size,
            num_left_chunks=num_left_chunks, 
        )  # (batch_size, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        
        # predictor_outs = self.calc_predictor(enc, enc_len)
        # encoder_out_mask = (make_valid_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index = self._predictor(
            encoder_out, None, mask=encoder_mask
        )
        pre_token_length = pre_token_length.round().long()
        
        # decoder_outs = self.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_outs = self._decoder.forward_by_embedding(
            encoder_out, encoder_mask, pre_acoustic_embeds, pre_token_length
        )
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1).to("cpu")
        pre_token_length = pre_token_length.to("cpu")
        
        beam_searcher = BeamSearchSequence(
            nbest=1, 
            vocab_size=decoder_out.size(-1), 
            beam_size=beam_size,
            beam_size_token=2*beam_size,
            
        )
        decoder_out_cpu = decoder_out.to("cpu")
        
        beam_search_results = beam_searcher.decode(emissions=decoder_out_cpu, lengths=pre_token_length)
        hyps = [result[0].tokens.tolist() for result in beam_search_results]
        return hyps
  
    ####################
    ##### property #####
    ####################
    @torch.jit.export
    def version(self) -> str:
        """ Export interface for c++ call, return version of the model
        """
        return self._model_version

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

