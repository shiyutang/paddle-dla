# --------------------------------------------------------
# Configurations for paddle-DLA
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------

import numpy as np
from easydict import EasyDict
import yaml

cfg = EasyDict()
# COMMON CONFIGS
cfg.name = '0413_paddle_DLA'
cfg.root = '/root/tpp/paddle/paddle_dla/'
cfg.nbr_class = 1000
# DATA
cfg.Data = EasyDict()
cfg.Data.Dir = '/root/tpp/paddle/data'
cfg.Data.dataset = 'imagenet'

# transforms
cfg.Trans = EasyDict()
cfg.Trans.crop_size = 224
cfg.Trans.scale_size = 256
cfg.Trans.crop_10 = True
cfg.Trans.down_ratio = 2
cfg.Trans.random_color = True
cfg.Trans.min_area_ratio = 0.08
cfg.Trans.aspect_ratio = 4.0 / 3

# Network
cfg.arch = 'dla34'

# Training
cfg.Train = EasyDict()
cfg.Train.epochs = 120
cfg.Train.start_epoch = 0
cfg.Train.lr = 0.1
cfg.Train.momentum = 0.9
cfg.Train.weight_decay = 1e-4
cfg.Train.resume = None  # ('', '')
cfg.Train.pretrained = None  # "imagenet"
cfg.Train.batch_size = 256
cfg.Train.num_workers = 0
cfg.Train.step_ratio = 0.1

# LOG
cfg.Log_print_freq = 10


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def yaml_load(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f)


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
