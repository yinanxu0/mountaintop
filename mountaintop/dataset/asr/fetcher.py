import random
import math
from pathlib import Path
from typing import List, Optional, Type, Union
import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.compliance.kaldi as kaldi
from PIL import Image
import numpy as np

from uphill import DocumentArray
from uphill.core.audio.augment import Resample, Speed
from uphill.core.text.vocab import Tokenizer, Vocabulary


from mountaintop.runx.logx import loggerx


FETCHER_MAP = {}
FETCHER_TRAIN_ONLY_MAP = {}

def register_fetcher(cls):
    cls_name = cls.__name__.lower().replace("fetcher", "")
    FETCHER_MAP[cls_name] = cls
    if hasattr(cls, "train_only") and cls.train_only == True:
        FETCHER_TRAIN_ONLY_MAP[cls_name] = cls
    return cls

def get_fetcher(name: str) -> Type[IterableDataset]:
    name = name.lower()
    if name not in FETCHER_MAP:
        loggerx.warning(
            f"{name} is not a valid fetcher, should be in "
            f"{FETCHER_MAP.keys()}")
        return None
    return FETCHER_MAP[name]

def is_train_fetcher(name: str) -> bool:
    if name in FETCHER_TRAIN_ONLY_MAP:
        return True
    return False

class BaseFetcher(IterableDataset):
    def __init__(self, data, *args, **kwargs):
        self._data = data
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        """ Return an iterator over the dataset processed by the
            given processor.
        """
        assert self._data is not None
        return self._iter_process()
    
    def set_epoch(self, epoch):
        self._data.set_epoch(epoch)
    
    def _iter_process(self):
        raise NotImplementedError("BaseFetcher._iter_process not implemented")


class PreFetcher(BaseFetcher):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example, "target")
            assert hasattr(example.source, "load_audio")
            waveform = example.source.load_audio()
            example.source.waveform = waveform
            yield example


@register_fetcher
class TokenizerFetcher(BaseFetcher):
    train_only = False
    def __init__(
        self,
        data,
        vocab_path: Union[Path, str],
        bpe_model: Optional[Union[Path, str]] = None,
        special_tokens: Optional[List[str]] = None,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        *args, **kwargs
    ):
        super().__init__(data, *args, **kwargs)
        self.vocab = Vocabulary(vocab_path=vocab_path)
        self.tokenizer = Tokenizer(
            bpe_model=bpe_model,
            special_tokens=special_tokens,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )

    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "target")
            assert hasattr(example.target, "load_text")
            text = example.target.load_text().strip()
            tokens = self.tokenizer.tokenize(text=text)
            token_ids = [self.vocab.token2id(token) for token in tokens]
            example.target.text = text
            example.target.tokens = tokens
            example.target.label = torch.tensor(token_ids)
            yield example


@register_fetcher
class FilterFetcher(BaseFetcher):
    """ Filter example according to feature and label length
        Inplace operation.

        Args::
            data: data manifests, `Iterable[uphill.Supervision]`
            src_max_length: drop utterance which is greater than max_length(10ms)
            src_min_length: drop utterance which is less than min_length(10ms)
            tgt_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            tgt_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[uphill.Supervision]
    """
    train_only = True
    def __init__(self, data, src_min_length=20, src_max_length=10240, 
           tgt_min_length=1, tgt_max_length=200,
           min_output_input_ratio=0.0005, max_output_input_ratio=1, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._src_min_length = src_min_length
        self._src_max_length = src_max_length
        self._tgt_min_length = tgt_min_length
        self._tgt_max_length = tgt_max_length
        self._min_output_input_ratio = min_output_input_ratio
        self._max_output_input_ratio = max_output_input_ratio
    
    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example, "target")
            assert hasattr(example.source, "waveform")
            assert hasattr(example.source, "sampling_rate")
            assert hasattr(example.target, "text")
            
            sampling_rate = example.source.sampling_rate
            if len(example.source.waveform.shape) == 1:
                num_src_samples = example.source.waveform.shape[0]
            else:
                num_src_samples = example.source.waveform.shape[1]
            num_src_frames = float(num_src_samples) / sampling_rate * 100
            
            tgt_length = len(example.target.text)
            if num_src_frames < self._src_min_length or num_src_frames > self._src_max_length \
                or tgt_length < self._tgt_min_length or tgt_length > self._tgt_max_length:
                loggerx.debug(f"Discard example({example.id}), src length: {num_src_frames}, tgt length: {tgt_length}")
                continue
            ratio = tgt_length*1.0 / num_src_frames
            if ratio < self._min_output_input_ratio or ratio > self._max_output_input_ratio:
                loggerx.debug(f"Discard example({example.id}), length ratio: {ratio}")
                continue
            yield example


@register_fetcher
class ResampleFetcher(BaseFetcher):
    train_only = False
    def __init__(self, data, sample_rate=16000, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._sample_rate = sample_rate
    
    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example.source, "waveform")
            assert hasattr(example.source, "sampling_rate")
            sample_rate = example.source.sampling_rate
            if sample_rate != self._sample_rate:
                waveform = example.source.waveform
                resampler = Resample(sample_rate, self._sample_rate)
                example.source.waveform = resampler(waveform)
            yield example


@register_fetcher
class SpeedFetcher(BaseFetcher):
    train_only = True
    def __init__(self, data, speeds: List[float] = None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._speeder_map = {}
        self._speeds = [0.9, 1.0, 1.1] if speeds is None else speeds
        for speed in self._speeds:
            if speed != 1.0:
                self._speeder_map[speed] = Speed(speed)
    
    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example.source, "waveform")
            assert hasattr(example.source, "sampling_rate")
            speed = random.choice(self._speeds)
            if speed != 1.0:
                sample_rate = example.source.sampling_rate
                waveform = example.source.waveform
                speeder = self._speeder_map[speed]
                example.source.waveform = speeder(waveform, sample_rate)
            yield example


@register_fetcher
class FeatureFetcher(BaseFetcher):
    train_only = False
    def __init__(self, data, type: str = "fbank", dim: int = 10, dither: float = 0.0,
     frame_length: int = 25, frame_shift: int = 10, num_ceps: int = 40, 
     high_freq: float = 0.0, low_freq: float = 20.0, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        assert type in ["fbank", "mfcc"]
        self._type = type
        self._dim = dim
        self._dither = dither
        self._frame_length = frame_length
        self._frame_shift = frame_shift
        self._num_ceps = num_ceps
        self._high_freq = high_freq
        self._low_freq = low_freq
    
    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example.source, "waveform")
            assert hasattr(example.source, "sampling_rate")
            waveform = example.source.waveform
            sample_rate = example.source.sampling_rate
            waveform = waveform * (1 << 15)
            # Only keep key, feat, label
            if self._type == "fbank": 
                featform = kaldi.fbank(torch.Tensor(waveform),
                                num_mel_bins=self._dim,
                                frame_length=self._frame_length,
                                frame_shift=self._frame_shift,
                                dither=self._dither,
                                energy_floor=1e-6,
                                sample_frequency=sample_rate)
            else:
                featform = kaldi.mfcc(torch.Tensor(waveform),
                                num_mel_bins=self._dim,
                                frame_length=self._frame_length,
                                frame_shift=self._frame_shift,
                                dither=self._dither,
                                num_ceps=self._num_ceps,
                                high_freq=self._high_freq,
                                low_freq=self._low_freq,
                                sample_frequency=sample_rate)
            example.source.waveform = None
            example.source.featform = featform
            yield example


@register_fetcher
class SpecAugFetcher(BaseFetcher):
    """ Do spectrum augmentation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask

        Returns
            Iterable[{key, feat, label}]
    """
    train_only = True
    def __init__(self, data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, time_wrap=False, max_w=80, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._num_t_mask = num_t_mask
        self._num_f_mask = num_f_mask
        self._max_t = max_t
        self._max_f = max_f
        
        # for future
        self._time_wrap = time_wrap
        self._max_w = max_w
    
    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example.source, "featform")
            
            featform = example.source.featform.clone().detach()
            max_frames = featform.size(0)
            max_freq = featform.size(1)
            # time warp
            if self._time_wrap and max_frames > self._max_w * 2:
                center = random.randrange(self._max_w, max_frames - self._max_w)
                warped = random.randrange(center - self._max_w, center + self._max_w) + 1
                left = Image.fromarray(featform[:center]).resize((max_freq, warped), Image.BICUBIC)
                right = Image.fromarray(featform[center:]).resize((max_freq, max_frames - warped), Image.BICUBIC)
                featform = np.concatenate((left, right), 0)
            # time mask
            for _ in range(self._num_t_mask):
                length = random.randint(1, min(self._max_t, max_frames-1))
                start = random.randint(0, max_frames - length)
                featform[start:start+length, :] = 0
            # freq mask
            for _ in range(self._num_f_mask):
                length = random.randint(1, self._max_f)
                start = random.randint(0, max_freq - length)
                featform[:, start:start+length] = 0 
            example.source.featform = featform
            yield example


@register_fetcher
class SpecSubFetcher(BaseFetcher):
    train_only = True
    def __init__(self, data, max_t=20, num_t_sub=3, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._max_t = max_t
        self._num_t_sub = num_t_sub
    
    def _iter_process(self):
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example.source, "featform")
            featform = example.source.featform.clone().detach()
            max_frames = featform.size(0)
            for _ in range(self._num_t_sub):
                length = random.randint(1, min(self._max_t, max_frames-1))
                start = random.randint(0, max_frames - length)
                end = start + length
                # only substitute the earlier time chosen randomly for current time
                offset = random.randint(0, start)
                featform[start:end, :] = featform[start - offset:end - offset, :]
            example.source.featform = featform
            yield example


@register_fetcher
class MixAugFetcher(BaseFetcher):
    train_only = True
    def __init__(
        self, 
        data, 
        aug_file: DocumentArray, 
        max_scale: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(data, *args, **kwargs)
        self.aug_data = DocumentArray.from_file(aug_file)
        self.max_scale = max_scale

    def _iter_process(self):
        """ batch the data """
        for example in self._data:
            assert hasattr(example, "source")
            assert hasattr(example.source, "waveform")
            if random.random() < 0.5:
                # to add mix audio augmentation
                def to_2d_form(waveform):
                    return np.expand_dims(waveform, axis=0) if len(waveform.shape) == 1 else waveform
                
                def get_wav_length(waveform):
                    return waveform.shape[0] if len(waveform.shape) == 1 else waveform.shape[1]
                
                # waveform shape: (num_channels, num_samples)    
                src_waveform = to_2d_form(example.source.waveform)
                src_waveform_length = get_wav_length(src_waveform)
                
                aug_doc = self.aug_data.sample(1)[0]
                aug_waveform = to_2d_form(aug_doc.load_audio())
                aug_waveform_length = get_wav_length(aug_waveform)

                if aug_waveform.shape[0] > 1:
                    ## conver multi channels to one channel: (1, num_samples)
                    aug_waveform = np.expand_dims(np.mean(aug_waveform, axis=0), axis=0)
                
                if src_waveform_length > aug_waveform_length:
                    # source longger than augment data
                    zoom_ratio = math.ceil(src_waveform_length/aug_waveform_length)
                    aug_waveform_reverse = np.expand_dims(aug_waveform[0][::-1], axis=0)
                    aug_waveform_obverse = aug_waveform
                    aug_waveform = aug_waveform_obverse
                    for i in range(zoom_ratio - 1):
                        current_waveform = aug_waveform if (i+1) == 0 else aug_waveform_reverse
                        aug_waveform = np.concatenate([aug_waveform, current_waveform], axis=1)
                # update augment waveform length
                aug_waveform_length = get_wav_length(aug_waveform)
                
                aug_start = random.randint(0, aug_waveform_length-src_waveform_length)
                aug_end = aug_start + src_waveform_length

                scale = random.randint(self.max_scale*50, self.max_scale*100)*1.0/100
                for channel in range(src_waveform.shape[0]):
                    src_waveform[channel] = src_waveform[channel]*scale + \
                        aug_waveform[0][aug_start:aug_end] * (1 - scale)

                example.source.waveform = src_waveform
            
            yield example


@register_fetcher
class BatchFetcher(BaseFetcher):
    train_only = False
    def __init__(self, data, batch_type: str = 'static', capcity: int = 32, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        assert batch_type in ["static", "dynamic"]
        self._type = batch_type
        self._capcity = capcity

    def _iter_process(self):
        """ batch the data """
        buffle = []
        if self._type == "static":
            """ Static batch the data by `capcity`"""
            for example in self._data:
                buffle.append(example)
                if len(buffle) >= self._capcity:
                    yield buffle
                    buffle = []
        else:
            """ Dynamic batch the data until the total frames in batch reach `capcity`"""
            max_frames_in_batch = 0
            for example in self._data:
                assert hasattr(example, "source")
                assert hasattr(example.source, "featform")
                assert isinstance(example.source.featform, torch.Tensor)
                current_sample_frames = example.source.featform.size(0)
                max_frames_in_batch = max(max_frames_in_batch, current_sample_frames)
                if max_frames_in_batch * (len(buffle)+1) > self._capcity:
                    yield buffle
                    buffle = []
                    max_frames_in_batch = current_sample_frames
                buffle.append(example)
        if len(buffle) > 0:
            yield buffle


class PaddingFetcher(BaseFetcher):
    def __init__(self, data, mode: str="train", *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        assert mode in ["train", "valid", "test"]
        self._mode = mode

    def _iter_process(self):
        for examples in self._data:
            assert isinstance(examples, list)
            
            keys = [example.id for example in examples]
            origin_feats = [example.source.featform for example in examples]
            feats_lengths = torch.tensor([example.source.featform.size(0) for example in examples], dtype=torch.int32)
            origin_labels = [example.target.label for example in examples]
            labels_lengths = torch.tensor([example.target.label.size(0) for example in examples], dtype=torch.int32)

            feats = pad_sequence(origin_feats,
                                        batch_first=True,
                                        padding_value=0)
            labels = pad_sequence(origin_labels,
                                        batch_first=True,
                                        padding_value=-1)
            # clear content to save memory
            for example in examples:
                example.source.waveform = None
                example.source.featform = None
                example.target.tokens = None
                example.target.label = None
            data_mapping = {
                "feat": feats,
                "feat_lengths": feats_lengths,
                "tgt": labels,
                "tgt_lengths": labels_lengths,
            }
            if self._mode == "test":
                data_mapping["keys"] = keys
            yield data_mapping
            


