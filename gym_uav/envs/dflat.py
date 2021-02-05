import collections
from collections import MutableMapping
import gym.spaces

def gflatten(d, parent_key='', sep='_'):
    items = []
    d2 = {k: d[k] for k in d}
    for k, v in d2.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping) or isinstance(v, gym.spaces.Dict):
            items.extend(gflatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

