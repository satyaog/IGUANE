#!/usr/bin/env python3
import argparse
import logging
import pathlib
import glob, re
import json
import sys
from iguane.log import logger
from iguane.fom import _CURRENT_FOM_VERSION, FIELDS, UGR_VERSIONS, RAWDATA, FOM, FOM_VERSIONS_WEIGHTS


def matchgpu(name, pat):
    if not pat:
        return True
    if not re.match('[*?\\[\\]]', pat) and name.startswith(pat):
        return True
    return glob.fnmatch.fnmatchcase(name.lower(), pat.lower())


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--reverse',   '-r', action='store_true',
                      help="Reverse listing")
    argp.add_argument('--sort',      '-s', action='store_true',
                      help="Sort GPU listing by unit")
    argp.add_argument('--list-units',      action='store_true',
                      help="List known Units")
    argp.add_argument('--list-ugr-versions', '--list-rgu-versions',
                      action='store_true',
                      help="List known UGR/RGU versions")
    argp.add_argument('--list-gpus', '-l', action='store_true',
                      help="List known GPUs")
    argp.add_argument('--dump-raw',        action='store_true',
                      help="Dump raw data")
    argp.add_argument('--input', '-i', type=pathlib.Path, default=None,
                      help="Input JSON file")
    argp.add_argument('--gpu',  '-G', type=str, default=None,
                      help="Selected GPU")
    argp.add_argument('--verbose', '-v',   action='count', default=0,
                      help="Increase verbosity level")

    ogrp = argp.add_argument_group('OUTPUT FORMAT', 'Output format control')
    ogrp.add_argument('--delimiter', '-d', type=str, default=',',
                      help="Delimiter for --parsable output")
    omtx = ogrp.add_mutually_exclusive_group()
    omtx.add_argument('--json',      '-j', action='store_true',
                      help="Dump output as JSON")
    omtx.add_argument('--parsable',  '-p', action='store_true',
                      help="Dump output as delimited text")

    ugrp = argp.add_argument_group('UNITS', 'Unit/FoM selection')
    ugrp.add_argument('--norm',            action='store_true',
                      help="Normalize the weights to 1.0")
    umtx = ugrp.add_mutually_exclusive_group()
    umtx.add_argument('--unit', '-u', '--fom', type=str.lower, default='ugr',
                      choices=sorted(set(FOM.keys()) | {'iguana', 'rgu'}),
                      help="Select unit-equivalence (FoM) to compute for every GPU")
    umtx.add_argument('--iguane',  action='store_const', dest='unit', const='iguane',
                      help="Select IGUANE/IGUANA unit-equivalence")
    umtx.add_argument('--iguana',  action='store_const', dest='unit', const='iguane',
                      help="Select IGUANE/IGUANA unit-equivalence")
    umtx.add_argument('--ugr',     action='store_const', dest='unit', const='ugr',
                      help="Select UGR/RGU unit-equivalence")
    umtx.add_argument('--rgu',     action='store_const', dest='unit', const='ugr',
                      help="Select UGR/RGU unit-equivalence")
    ugrp.add_argument('--fom-version', type=str, default=_CURRENT_FOM_VERSION,
                      choices=FOM_VERSIONS_WEIGHTS.keys(),
                      help="Select Figure-of-Merit ponderation version")
    ugrp.add_argument('--custom-weights', type=str,
                      help='Use custom weights in the form \'{"ref": "GPU", ' + ', '.join(f'"{f}": 0.0' for f in FIELDS) + '}\'')

    # Parse Arguments
    args = argp.parse_args(sys.argv[1:])
    args.unit = {'iguana': 'iguane', 'rgu': 'ugr'}.get(args.unit, args.unit)
    if args.fom_version:
        args.unit = "fom_version"
    if args.verbose:
        logger.setLevel(logging.CRITICAL - (args.verbose - 1) * 10)

    if   args.list_units:
        units = list(FOM.keys())
        if args.reverse:
            units = units[::-1]
        if args.json:
            print(json.dumps(units, indent=2))
        else:
            for k in units:
                print(k)
    elif args.list_gpus:
        gpus = RAWDATA.keys()
        if args.reverse:
            gpus = reversed(gpus)
        gpus = [k for k in gpus if matchgpu(k, args.gpu)]
        if args.json:
            print(json.dumps(gpus, indent=2))
        else:
            for k in gpus:
                print(k)
    elif args.list_ugr_versions:
        ugr_versions = UGR_VERSIONS.keys()
        if args.reverse:
            ugr_versions = reversed(ugr_versions)
        ugr_versions = {k:UGR_VERSIONS[k] for k in ugr_versions}
        if args.json:
            print(json.dumps(ugr_versions, indent=2))
        else:
            for k in ugr_versions.keys():
                print(k) # Only print name, not weights data
    elif args.dump_raw:
        print(json.dumps(RAWDATA, indent=2))
    elif args.input:
        with open(args.input) as f:
            CLUSTER = json.load(f)
        CLUSTER = {
            k: c*FOM[args.unit](k, args=args)
            for k, c in CLUSTER.items()
            if  k    in RAWDATA
        }
        gpus = CLUSTER.keys()
        if args.reverse:
            gpus = reversed(gpus)
        gpus = [k for k in gpus if matchgpu(k, args.gpu)]
        if args.sort:
            gpus = sorted(gpus, key=lambda k:CLUSTER[k], reverse=args.reverse)
        CLUSTER = {k:CLUSTER[k] for k in gpus}
        CLUSTER = {
            "breakdown": CLUSTER,
            "total":     sum(CLUSTER.values(), 0),
        }
        if  not CLUSTER["breakdown"]:
            del CLUSTER["breakdown"]
        if args.json:
            print(json.dumps(CLUSTER, indent=2))
        else:
            # Just the total units of GPU
            if isinstance(CLUSTER["total"], int):
                print(CLUSTER["total"])
            else:
                print(f'{CLUSTER["total"]:5.2f}')
    else:
        DATA = {}
        gpus = RAWDATA.keys()
        if args.reverse:
            gpus = reversed(gpus)
        gpus = [k for k in gpus if matchgpu(k, args.gpu)]
        for k in gpus:
            DATA[k] = FOM[args.unit](k, args=args)
        if args.sort:
            gpus = sorted(gpus, key=lambda k:DATA[k], reverse=args.reverse)
        if args.json:
            print(json.dumps({k:DATA[k] for k in gpus}, indent=2))
        else:
            for k in gpus:
                if args.parsable:
                    print(f"{k}{args.delimiter}{DATA[k]}")
                else:
                    print(f"{DATA[k]:-5.2f} {k}")

    sys.exit(0)
