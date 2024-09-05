import copy
import importlib.resources
import json
try:
    import tomllib
except ModuleNotFoundError:
    # for Python before 3.11
    import pip._vendor.tomli as tomllib

from iguane.log import logger

# Import RAWDATA (a dictionary mapping GPU name to GPU flops and data) from a resource file
RAWDATA = tomllib.loads(importlib.resources.read_text('iguane', 'rawdata.toml'))
FIELDS = [
    k for k, v in RAWDATA["K80"].items()
    if isinstance(v, (int, float))
]

FOM_VERSIONS_WEIGHTS = {
    # v1.0 is equivalent to RGU
    # https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling
    '1.0':    { 'ref': 'A100-SXM4-40GB', 'fp16': 1.6, 'fp32': 1.6,              'memgb': 0.8                 },
    'ugr':    { 'ref': 'A100-SXM4-40GB', 'fp16': 1.6, 'fp32': 1.6,              'memgb': 0.8                 },
    '2.0-0':  { 'ref': 'A100-SXM4-80GB', 'fp16': 0.2, 'fp32': 0.1, 'tf32': 0.2, 'memgb': 0.25, 'membw': 0.25 },
    'iguane': { 'ref': 'A100-SXM4-80GB', 'fp16': 0.2, 'fp32': 0.2, 'tf32': 0.2, 'memgb': 0.2,  'membw': 0.2  },
}

_CURRENT_FOM_VERSION = list(
    reversed(
        sorted((k for k in FOM_VERSIONS_WEIGHTS.keys() if k.replace(".", "").isdigit()))
    )
)[0]

UGR_VERSIONS = {
    # v1.0 is equivalent to UGR
    # https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling
    '1.0':        { 'fp16': 1.6, 'fp32': 1.6, 'memgb': 0.8 },
    '1.0-renorm': { 'fp16': 0.4, 'fp32': 0.4, 'memgb': 0.2 },
}


def fom(f):
    globals().setdefault('FOM', {})
    assert f.__name__.startswith('fom_') and f.__name__ != 'fom_', \
           'Figure-of-Merit function names must start with "fom_"!'
    FOM[f.__name__[4:]] = f
    return f


@fom
def fom_count(name, *, args=None):
    return 1


@fom
def fom_fp16(name, *, args=None):
    data = RAWDATA[name]
    return data['fp16'] or data['fp32']


@fom
def fom_fp32(name, *, args=None):
    return RAWDATA[name]['fp32']


@fom
def fom_fp64(name, *, args=None):
    return RAWDATA[name]['fp64']


@fom
def fom_tf32(name, *, args=None):
    data = RAWDATA[name]
    return data['tf32'] or data['fp32']


@fom
def fom_ugr(name, *, args=None):
    args = copy.copy(args)
    args.fom_version = 'ugr'
    return fom_fom_version(name, args=args)


@fom
def fom_iguane(name, *, args=None):
    args = copy.copy(args)
    args.fom_version = 'iguane'
    return fom_fom_version(name, args=args)


@fom
def fom_fom_version(name, *, args=None):
    return fom_custom_weights(name, args=args)


@fom
def fom_custom_weights(name, *, args=None):
    fom_version = args.fom_version if args else _CURRENT_FOM_VERSION
    weights = args.custom_weights if args and args.custom_weights else FOM_VERSIONS_WEIGHTS[fom_version].copy()
    if isinstance(weights, str):
        weights = json.loads(weights)

    ref  = RAWDATA[weights.pop('ref')]
    data = RAWDATA[name].copy()
    data['tf32'] = data['tf32'] or data['fp32']
    data['fp16'] = data['fp16'] or data['fp32']

    if args.norm:
        _sum = sum(weights.values())
        weights = {k: v / _sum for k, v in weights.items()}

    _sum = 0
    for k, w in weights.items():
        ratio = w * data[k] / ref[k]
        _sum += ratio
        logger.debug(f"{k}\t{ratio:.4f}\t{name}")
    return _sum

