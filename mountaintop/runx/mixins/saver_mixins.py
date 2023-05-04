import os
from typing import Dict
import torch

from mountaintop.runx.logx import loggerx

class SaverMixins():
    # def save_checkpoint(self, eval_metrics, *arg, **kwargs):
    #     """checkpoint saving."""
    #     if self._eval_metric not in eval_metrics:
    #         raise ValueError(
    #             f"The model's metric {self._eval_metric} is not available!")

    #     metric = eval_metrics[self._eval_metric]
    #     self._best_metric = (
    #         max(self.best_metric, metric) if self._higher_better else min(
    #             self.best_metric, metric))

    #     loggerx.save_model(
    #         self.state_dict(),
    #         metric=metric,
    #         epoch=self.epoch,
    #         higher_better=self._higher_better,
    #         delete_old=not self._keep_old_model
    #     )
    
    def load_checkpoint(
        self, 
        filepath: str, 
        device: torch.device = torch.device("cpu")
    ):
        """Restore a model and return a dict with any meta data included in the
        snapshot."""
        checkpoint = None
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            info_str = f"=> loaded checkpoint '{filepath}'"
            info_str += f" (epoch {checkpoint['__epoch']})" if "__epoch" in checkpoint else ""
            loggerx.info(info_str)
            # self.load_state_dict(checkpoint)
        else:
            loggerx.warning(f"=> no checkpoint found at '{filepath}'")
            raise FileNotFoundError(f'checkpoint file {filepath} not found!')
        return checkpoint
    
    def load_average_checkpoints(
        self, 
        filepath_start: str,
        filepath_end: str,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """Average model parameters over the range with given
        start model (excluded) and end model.

        Let start = current_steps of model-start;
            end = current_steps of model-end;
            interval = end - start.
        Then the average model over range from start (excluded) to end is
        (1) avg = (model_end * end - model_start * start) / interval.
        It can be written as
        (2) avg = model_end * weight_end + model_start * weight_start,
            where weight_end = end / interval,
                weight_start = -start / interval = 1 - weight_end.
        Since the terms `weight_end` and `weight_start` would be large
        if the model has been trained for lots of batches, which would cause
        overflow when multiplying the model parameters.
        To avoid this, we rewrite (2) as:
        (3) avg = (model_end + model_start * (weight_start / weight_end))
                * weight_end

        The model index could be epoch number or iteration number.

        Args:
        filename_start:
            Checkpoint filename of the start model. We assume it
            is saved by :func:`save_checkpoint`.
        filename_end:
            Checkpoint filename of the end model. We assume it
            is saved by :func:`save_checkpoint`.
        device:
            Move checkpoints to this device before averaging.
        """
        state_dict_start = torch.load(filepath_start, map_location=device)
        state_dict_end = torch.load(filepath_end, map_location=device)
        
        step_key = "__step"
        average_key = "__average_model"
        assert step_key in state_dict_start and step_key in state_dict_end
        assert average_key in state_dict_start and average_key in state_dict_end
        
        start_steps = state_dict_start[step_key]
        end_steps = state_dict_end[step_key]
        interval = end_steps - start_steps
        assert interval > 0, f"interval of steps should be positive not {interval}"
        
        weight_end = end_steps / interval
        weight_start = 1 - weight_end
        
        model_end = state_dict_end[average_key]
        model_start = state_dict_start[average_key]
        avg = model_end

        # scale the weight to avoid overflow
        self.average_state_dict(
            state_dict_1=avg,
            state_dict_2=model_start,
            weight_1=1.0,
            weight_2=weight_start / weight_end,
            scaling_factor=weight_end,
        )

        return avg

    def average_state_dict(
        self,
        state_dict_1: Dict[str, torch.Tensor],
        state_dict_2: Dict[str, torch.Tensor],
        weight_1: float,
        weight_2: float,
        scaling_factor: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Average two state_dict with given weights:
        state_dict_1 = (state_dict_1 * weight_1 + state_dict_2 * weight_2)
        * scaling_factor
        It is an in-place operation on state_dict_1 itself.
        """
        # Identify shared parameters. Two parameters are said to be shared
        # if they have the same data_ptr
        uniqued: Dict[int, str] = dict()
        for k, v in state_dict_1.items():
            v_data_ptr = v.data_ptr()
            if v_data_ptr in uniqued:
                continue
            uniqued[v_data_ptr] = k

        uniqued_names = list(uniqued.values())
        for k in uniqued_names:
            if not (state_dict_1[k].dtype in [torch.float16, torch.float32, torch.float64]):
                continue
            state_dict_1[k] *= weight_1
            state_dict_1[k] += (
                state_dict_2[k].to(device=state_dict_1[k].device) * weight_2
            )
            state_dict_1[k] *= scaling_factor


