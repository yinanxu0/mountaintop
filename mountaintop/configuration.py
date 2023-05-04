from mountaintop.runx.logx import loggerx
from mountaintop.resources import get_resources


def get_model_configuration(
    model_name: str,
    **kwargs,
):
    assert "/" in model_name, f"/ should in model_name(value: {model_name})"
    parent_model_name, sub_model_name = model_name.split("/")
    if "-" in sub_model_name :
        sub_model_name, model_size = sub_model_name.split("-")
    else:
        model_size = "base"
    
    full_model_name = f"{sub_model_name}-{model_size}"
    configuration = get_resources(
        type="model", 
        parent_name = parent_model_name,
        sub_name = full_model_name,
    )
    return configuration


def get_fetcher_configuration(
    model_name: str,
    **kwargs,
):
    assert "/" in model_name, f"/ should in model_name(value: {model_name})"
    parent_model_name, sub_model_name = model_name.split("/")
    
    configuration = get_resources(
        type="fetcher", 
        parent_name = parent_model_name,
        sub_name = sub_model_name,
    )
    return configuration

