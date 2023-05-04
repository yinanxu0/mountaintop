import os
import json

from mountaintop.utils.yaml import load_yaml
from mountaintop.runx.logx import loggerx


def get_resources(
    type: str,
    parent_name: str,
    sub_name: str,
):  
    assert type in ["model", "fetcher"]
    resources_dir = os.path.dirname(os.path.realpath(__file__))
    resource_name = os.path.join(resources_dir, f"{type}_config/{parent_name}", sub_name)
    data = {}
    if os.path.exists(resource_name + ".yaml"):
        resource_path = resource_name + ".yaml"
        data = load_yaml(resource_path)
    elif os.path.exists(resource_name + ".json"):
        resource_path = resource_name + ".json"
        content = open(resource_path, "r").read()
        data = json.loads(content)
    else:
        loggerx.warning(f"{resource_name} not exists!!! Please double check.")
        
    return data