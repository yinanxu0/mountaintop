import os
import sys
import warnings
import importlib
import itertools
from pathlib import Path
from types import ModuleType
from typing import Optional, Union
from typing import Optional
from functools import wraps


from mountaintop.runx.logx import loggerx


def import_module(module_path: str):
    pkg_name, cls_name = module_path.split(':')
    try:
        pkg = importlib.import_module(pkg_name)
    except ModuleNotFoundError as _:
        pkg_path = os.path.join(*pkg_name.rstrip('.py').split('.')) + '.py'
        pkg = import_pkg(pkg_path)
    loggerx.debug(f"dynamic import: from {pkg_name} import {cls_name}")
    module_cls = getattr(pkg, cls_name)
    return module_cls


def import_pkg(path: Union[str, Path]) -> ModuleType:
    """Imports and returns a module from the given path, which can be a file (a
    module) or a directory (a package)."""
    path = Path(str(path))

    if not path.exists():
        raise ImportError(path)

    pkg_path = resolve_package_path(path)
    if pkg_path is not None:
        pkg_root = pkg_path.parent
        names = list(path.with_suffix('').relative_to(pkg_root).parts)
        if names[-1] == '__init__':
            names.pop()
        module_name = '.'.join(names)
    else:
        pkg_root = path.parent
        module_name = path.stem

    # change sys.path permanently
    if str(pkg_root) != sys.path[0]:
        sys.path.insert(0, str(pkg_root))
    loggerx.info(module_name)
    importlib.import_module(module_name)
    mod = sys.modules[module_name]

    return mod


def resolve_package_path(path: Path) -> Optional[Path]:
    """Return the Python package path by looking for the last directory upwards
    which still contains an __init__.py.

    Return None if it can not be determined.
    """
    result = None
    for parent in itertools.chain((path, ), path.parents):
        if parent.is_dir():
            if not parent.joinpath('__init__.py').is_file():
                break
            if not parent.name.isidentifier():
                break
            result = parent
    return result


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


################
## decorators ##
################
def requires_module(*modules: str):
    """Decorate function to give error message if invoked without required optional modules.
    This decorator is to give better error message to users rather
    than raising ``NameError:  name 'module' is not defined`` at random places.
    """
    missing = [m for m in modules if not is_module_available(m)]

    if not missing:
        # fall through. If all the modules are available, no need to decorate
        def decorator(func):
            return func
    else:
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f'{func.__module__}.{func.__name__} requires {", ".join(missing)}')
            return wrapped
    return decorator


def deprecated(direction: str, version: Optional[str] = None):
    """Decorator to add deprecation message
    Args:
        direction (str): Migration steps to be given to users.
        version (str or int): The version when the object will be removed
    """
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            message = (
                f'{func.__module__}.{func.__name__} has been deprecated '
                f'and will be removed from {"future" if version is None else version} release. '
                f'{direction}')
            warnings.warn(message, stacklevel=2)
            # loggerx.warning(message)
            return func(*args, **kwargs)
        return wrapped
    return decorator