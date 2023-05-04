import torch


from mountaintop import __version__
from mountaintop.runx.logx import loggerx
from mountaintop.utils.git import git_version
from mountaintop.core.internal.module import import_module


#### Base Model Interface ####
class ModelInterface(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model_version = f"{__version__}+{git_version(short=True)}"
    
    #####################
    ##### interface #####
    #####################
    def forward(self, *args, **kwargs):
        error_msg = f"forward method not implemented in {self.__class__}"
        if len(args) > 0:
            error_msg += f"\n\targs: {args}"
        if len(kwargs) > 0:
            error_msg += f"\n\tkwargs: {kwargs}"
        raise NotImplementedError(error_msg)

    ####################
    ##### property #####
    ####################
    @torch.jit.export
    def version(self) -> str:
        """ Export interface for c++ call, return version of the model
        """
        return self._model_version

    ##################
    ##### method #####
    ##################
    @classmethod
    def create_from_config(cls, configs):
        model_dict = {}
        def _init_model(configs):
            module_dict = {}
            keys = list(configs.keys())
            for key in keys:
                value = configs.get(key)
                if "module" not in value:
                    continue
                configs.pop(key)
                loggerx.info(f"Init {key} from {value['module']}")
                submodule = import_module(value["module"])
                value.pop("module")
                module_dict[key] = submodule(**value)                
            return module_dict
        
        def _init_loss(configs):
            loss_funcs = {}
            loss_weight = {}
            keys = list(configs.keys())
            for key in keys:
                value = configs.pop(key)
                if "weight" in key:
                    loss_weight[key] = value
                else:
                    if "module" not in value:
                        continue
                    submodule = import_module(value["module"])
                    value.pop("module")
                    loss_funcs[key] = submodule(**value)  
            return loss_funcs, loss_weight
        model_dict = _init_model(configs=configs)
        loss_funcs, loss_weight = _init_loss(configs=configs.pop("loss"))
        return cls(
            **model_dict,
            **configs["base_config"],
            loss_funcs=loss_funcs,
            loss_weight=loss_weight
        )
    
    
    


