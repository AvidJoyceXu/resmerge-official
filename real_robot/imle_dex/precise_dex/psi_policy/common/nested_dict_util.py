import functools
from typing import Any, Tuple

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


def obs_keys_by_type(shape_meta: Any, obs_type: str = "rgb") -> Tuple[str, ...]:
    """Select shape_meta observation keys by type, preserving YAML declaration order."""
    if shape_meta is None:
        return ()
    if DictConfig is not None and isinstance(shape_meta, DictConfig):
        shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
    obs = shape_meta.get("obs") if isinstance(shape_meta, dict) else None
    if not obs:
        return ()
    out = []
    for k, v in obs.items():
        if isinstance(v, dict) and v.get("type") == obs_type:
            out.append(k)
    return tuple(out)


def nested_dict_map(f, x):
    """
    Map f over all leaf of nested dict x
    """

    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y

def nested_dict_reduce(f, x):
    """
    Map f over all values of nested dict x, and reduce to a single value
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result
