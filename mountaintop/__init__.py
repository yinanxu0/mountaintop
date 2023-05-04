from mountaintop.runx.logx import loggerx

from . import bin
from . import core
from . import dataset
from . import layers
from . import models
from . import runx
from . import utils



def get_package_version():
    import os
    version_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")
    version = '0.0.0'
    for content in open(version_file, 'r', encoding='utf8').readlines():
        content = content.strip()
        if len(content) > 0:
            version = content
            break
    return version

__version__ = get_package_version()