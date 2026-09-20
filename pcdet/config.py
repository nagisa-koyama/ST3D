from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if isinstance(value, tuple):
            value = list(value)

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def _load_base_yaml(base_path):
    """Load a `_BASE_CONFIG_` file, mirroring cfg_from_yaml_file's loader fallback."""
    with open(base_path, 'r') as f:
        print("{} is loaded".format(base_path))
        try:
            return yaml.full_load(f)
        except Exception:
            f.seek(0)
            return yaml.safe_load(f)


def _fill_missing_from_base(child, base, _chain=()):
    """Recursively fill keys from a `_BASE_CONFIG_` file into `child`, WITHOUT overwriting any
    key `child` already explicitly defines. This implements proper base-config inheritance
    (child values always take precedence over the base's); the base only supplies defaults for
    keys the child doesn't set itself.

    A base file may itself declare `_BASE_CONFIG_`. Such chains are expanded to arbitrary depth,
    with the nearest definition winning at every level. Expanding only one level was a real
    regression between `de9f9d7` (2026-08-23) and this fix: configs reaching their dataset via
    `DATA_CONFIGS.<NAME>._BASE_CONFIG_` -> another config with its own `_BASE_CONFIG_` silently
    lost the whole second hop (`DATA_PROCESSOR`, `POINT_CLOUD_RANGE`, ...). See
    experiments_md/20260921_01_base_config_recursion_regression_fix.md.
    """
    if '_BASE_CONFIG_' in base:
        base_path = base['_BASE_CONFIG_']
        if base_path in _chain:
            raise ValueError('Cyclic _BASE_CONFIG_ chain: {}'.format(
                ' -> '.join(list(_chain) + [base_path])))
        _fill_missing_from_base(base, _load_base_yaml(base_path), _chain + (base_path,))

    for key, val in base.items():
        if key == '_BASE_CONFIG_':
            # Bookkeeping key only. `child` keeps its own, so a resolved config records the file
            # it actually declared rather than an inherited ancestor's path.
            continue
        if key not in child:
            child[key] = val
        elif isinstance(val, dict) and isinstance(child[key], dict):
            _fill_missing_from_base(child[key], val, _chain)
    return child


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        base_path = new_config['_BASE_CONFIG_']
        _fill_missing_from_base(new_config, _load_base_yaml(base_path), (base_path,))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    print("{} is loaded".format(cfg_file))
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.full_load(f)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
