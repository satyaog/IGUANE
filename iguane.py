#!/usr/bin/env python3
import argparse
import glob, re
import json
import sys


def matchgpu(name, pat):
    if not pat:
        return True
    if not re.match('[*?\\[\\]]', pat) and name.startswith(pat):
        return True
    return glob.fnmatch.fnmatchcase(name.lower(), pat.lower())
def fom(f):
    globals().setdefault('FOM', {})
    assert f.__name__.startswith('fom_') and f.__name__ != 'fom_', \
           'Figure-of-Merit function names must start with "fom_"!'
    FOM[f.__name__[4:]] = f
    return f


#
# FP64 is scalar FP64 throughput.
# FP32 is scalar FP32 throughput.
# FP16 is:
#   - Tensor Core FP16    throughput @ 0% sparsity,  if Tensor Cores supported;
#   - Packed FP16         throughput,                if Tensor Cores unsupported but packed FP16 supported;
#   - Scalar FP16         throughput,                if neither Tensor Cores nor packed FP16 supported but FP16 natively supported;
#   - Scalar FP32         throughput,                if neither Tensor Cores nor packed FP16 nor scalar FP16 supported.
# TF32 is:
#   - Tensor Core TF32    throughput @ 0% sparsity,  if Tensor Core TF32 datatype supported;
#   - Scalar FP32         throughput,                if Tensor Core TF32 datatype not supported, or
#   - Throughput @ 0% sparsity (100% dense) of some easy-to-enable intermediate precision between FP16 and FP32, if superior to FP32.
# MEMGB is capacity  of GPU in gigabytes.
# MEMBW is bandwidth of GPU to global memory, in gigabytes/s.
# TDP   is Thermal Design Power, the expected maximum power dissipation of GPU in Watts.
#
#
# https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-datasheet.pdf
#     P100 SXM2:  Die GP100, 93.3% enabled (56 SMs, no Tensor Cores) @ Boost Clock 1480 MHz
#                 FP32: 1480e6*3584*2/1e12        =  10.60864  TFLOPS (1 FP32 FMA per CUDA Core)
#                 FP64: 1480e6*3584*2*0.5/1e12    =   5.30432  TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                 FP16: 1480e6*3584*2*2/1e12      =  21.21728  TFLOPS (2 FP16 FMA per CUDA Core, 2:1 of FP32; Packed arithmetic; No Tensor Cores; No gains from sparsity available)
#     P100 PCIe:  Die GP100, 93.3% enabled (56 SMs, no Tensor Cores) @ Boost Clock 1303 MHz
#                 FP32: 1303e6*3584*2/1e12        =   9.339904 TFLOPS (1 FP32 FMA per CUDA Core)
#                 FP64: 1303e6*3584*2*0.5/1e12    =   4.669952 TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                 FP16: 1303e6*3584*2*2/1e12      =  18.679808 TFLOPS (2 FP16 FMA per CUDA Core, 2:1 of FP32; Packed arithmetic; No Tensor Cores; No gains from sparsity available)
#
# https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
# https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
#     V100 SXM2:  Die GV100-895-A1 (16GB) / GV100-896-A1 (32GB), 95.2% enabled (80 SMs x 8 1st-gen Tensor Cores) @ Boost Clock 1530 MHz
#                 FP32: 1530e6*5120*2/1e12        =  15.6672  TFLOPS (1 FP32 FMA per CUDA Core)
#                 FP64: 1530e6*5120*2*0.5/1e12    =   7.8336  TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                 FP16: 1530e6*640*(4*4*4)*2/1e12 = 125.3376  TFLOPS (640 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle; no gains from sparsity available)
#     V100 PCIe:  Die GV100-893-A1 (16GB) / GV100-897-A1 (32GB), 95.2% enabled (80 SMs x 8 1st-gen Tensor Cores) @ Boost Clock 1370 MHz
#                 FP32: 1370e6*5120*2/1e12        =  14.0288  TFLOPS (1 FP32 FMA per CUDA Core)
#                 FP64: 1370e6*5120*2*0.5/1e12    =   7.0144  TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                 FP16: 1370e6*640*(4*4*4)*2/1e12 = 112.2304  TFLOPS (640 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle; no gains from sparsity available)
#     V100S PCIe: Die GV100-907A-A1, 95.2% enabled (80 SMs x 8 1st-gen Tensor Cores) @ Boost Clock 1597 MHz
#                 FP32: 1597e6*5120*2/1e12        =  16.35328 TFLOPS (1 FP32 FMA per CUDA Core)
#                 FP64: 1597e6*5120*2*0.5/1e12    =   8.17664 TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                 FP16: 1597e6*640*(4*4*4)*2/1e12 = 130.82624 TFLOPS (640 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle; no gains from sparsity available)
#
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
# https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
#     A100 PCIe & SXM4: Die GA100 (PCIe 40GB: GA100-883AA-A1), 84.375% enabled (108 SMs x 4 3rd-gen Tensor Cores) @ Boost Clock 1410 MHz
#                       FP32: 1410e6*6912*2/1e12        =  19.49184 TFLOPS (1 FP32 FMA per CUDA Core)
#                       FP64: 1410e6*6912*2*0.5/1e12    =   9.74592 TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                       FP16: 1410e6*432*(8*4*8)*2/1e12 = 311.86944 TFLOPS (432 Tensor Cores contracting 8x4x8 FP16 FMAs/clock cycle @ 0% sparsity)
#                       TF32: 1410e6*432*(8*4*4)*2/1e12 = 155.93472 TFLOPS (432 Tensor Cores contracting 8x4x4 TF32 FMAs/clock cycle @ 0% sparsity, 1:2 of FP16)
#
# https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
# https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
#     A6000: Die GA102, fully enabled (84 SMs x 4 3rd-gen Tensor Cores) @ Boost Clock 1800 MHz
#            FP32: 1800e6*10752*2/1e12       =  38.7072 TFLOPS (1 FP32 FMA per CUDA Core)
#            FP64: 1800e6*84*2*2/1e12        =   0.6048 TFLOPS (2 FP64 FMA ALUs per SM, 1:64 of FP32)
#            FP16: 1800e6*336*(4*4*8)*2/1e12 = 154.8288 TFLOPS (336 Tensor Cores contracting 4x4x8 FP16 FMAs/clock cycle @ 0% sparsity)
#            FP16: 1800e6*336*(4*4*4)*2/1e12 =  77.4144 TFLOPS (336 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle @ 0% sparsity, 1:2 of FP16, 2:1 of FP32)
#
# https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-8000-us-nvidia-946977-r1-web.pdf
# https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
#     RTX8000: Die TU102, fully enabled (72 SMs x 8 2nd-gen Tensor Cores) @ Boost Clock 1770 MHz
#              FP32: 1770e6*4608*2/1e12        =  16.31232 TFLOPS (1 FP32 FMA per CUDA Core)
#              FP64: 1770e6*72*2*2/1e12        =   0.50976 TFLOPS (2 FP64 FMA ALUs per SM, 1:32 of FP32)
#              FP16: 1770e6*576*(4*4*4)*2/1e12 = 130.49856 TFLOPS (576 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle; no gains from sparsity available)
#     T4:      Die TU104, 83.3% enabled (40 SMs x 8 2nd-gen Tensor Cores) @ Boost Clock 1590 MHz
#              FP32: 1590e6*2560*2/1e12        =   8.1408  TFLOPS (1 FP32 FMA per CUDA Core)
#              FP64: 1590e6*40*2*2/1e12        =   0.2544  TFLOPS (2 FP64 FMA ALUs per SM, 1:32 of FP32)
#              FP16: 1590e6*320*(4*4*4)*2/1e12 =  65.1264  TFLOPS (320 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle; no gains from sparsity available)
#
# https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
# https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413
#     L40S: Die AD102, 98.6% enabled (142 SMs x 4 = 568 4th-gen Tensor Cores) @ Boost Clock 2520 MHz
#           L40S datasheet is internally inconsistent:
#             - FP16/BF16 @ 0% sparsity are computed as 362.50 but that is not 50% of 733.
#               - 50% of 733 is 366.50 and:
#                 - "366" is the declared TF32 with 50% sparsity
#                 - 366/2 = 183 is the declared peak TF32 with 0% sparsity
#                 - 91.6*2 = ~183 and 91.6 is the declared FP32 performance
#                 - 733 is the declared peak FP8/INT8/INT4 @ 0% sparsity
#                 - 733*2 = 1466 is the declared peak FP8/INT8/INT4 @ 50% sparsity
#           FP32: 2520e6*18176*2/1e12       =  91.60704 TFLOPS (1 FP32 FMA per CUDA Core)
#           FP64: 2520e6*142*2*2/1e12       =   1.43136 TFLOPS (2 FP64 FMA ALUs per SM, 1:64 of FP32, verified empirically)
#           FP16: 2520e6*568*(4*4*8)*2/1e12 = 366.42816 TFLOPS (568 Tensor Cores contracting 4x4x8 FP16 FMAs/clock cycle @ 0% sparsity)
#           TF32: 2520e6*568*(4*4*4)*2/1e12 = 183.21408 TFLOPS (568 Tensor Cores contracting 4x4x4 FP16 FMAs/clock cycle @ 0% sparsity, 1:2 of FP16, 2:1 of FP32)
#
# https://resources.nvidia.com/en-us-tensor-core
# https://www.nvidia.com/en-us/data-center/h100/
#     H100_SXM has 80GB memory and H100_NVL has 94GB (warning, they provide the "cheating" values with sparsity)
#     H100 SXM5: Die GH100, 91.6% enabled (132 SMs x 4 = 528 4th-gen Tensor Cores) @ Boost Clock 1830 MHz (FP8, FP16, BF16, TF32 Tensor),
#                                                                                                1980 MHz (FP32 non-Tensor, FP64)
#                FP32: 1980e6*16896*2/1e12        =  66.90816 TFLOPS (1   FP32 FMA per CUDA Core)
#                FP64: 1980e6*16896*2*0.5/1e12    =  33.45408 TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                FP16: 1830e6*528*(8*4*16)*2/1e12 = 989.42976 TFLOPS (528 Tensor Cores contracting 8x4x16 FP16 FMAs/clock cycle @ 0% sparsity)
#                TF32: 1830e6*528*(8*4*8)*2/1e12  = 494.71488 TFLOPS (528 Tensor Cores contracting 8x4x8  TF32 FMAs/clock cycle @ 0% sparsity, 1:2 of FP16)
#     H100 PCIe: Die GH100, 79.2% enabled (114 SMs x 4 = 456 4th-gen Tensor Cores) @ Boost Clock 1620 MHz (FP8, FP16, BF16, TF32 Tensor),
#                                                                                                1755 MHz (FP32 non-Tensor, FP64)
#                FP32: 1755e6*14592*2/1e12        =  51.21792 TFLOPS (1   FP32 FMA per CUDA Core)
#                FP64: 1755e6*14592*2*0.5/1e12    =  25.60896 TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                FP16: 1620e6*456*(8*4*16)*2/1e12 = 756.44928 TFLOPS (456 Tensor Cores contracting 8x4x16 FP16 FMAs/clock cycle @ 0% sparsity)
#                TF32: 1620e6*456*(8*4*8)*2/1e12  = 378.22464 TFLOPS (456 Tensor Cores contracting 8x4x8  TF32 FMAs/clock cycle @ 0% sparsity, 1:2 of FP16)
#     H100 NVL:  Die GH100, () @ Boost Clock 1785 MHz (?)
#                (Aucune information sur géométrie et clocks)
#                (Inférence 1: 114 SMs x 4 = 456 4th-gen Tensor Cores @ Boost Clock 1789 MHz (FP8, FP16, BF16, TF32 Tensor),
#                                                                                   2056 MHz (FP32 non-Tensor, FP64))
#                (OU:)
#                (Inférence 2: 132 SMs x 4 = 528 4th-gen Tensor Cores @ Boost Clock 1545 MHz (FP8, FP16, BF16, TF32 Tensor),
#                                                                                   1785 MHz (FP32 non-Tensor, FP64))
#                (La seconde inférence est plus plausible car cohérente avec le Max Boost Clock déclaré dans Product Brief)
#                FP32: 1785e6*16896*2/1e12        =  60.31872 TFLOPS (1   FP32 FMA per CUDA Core)
#                FP64: 1785e6*16896*2*0.5/1e12    =  30.15936 TFLOPS (0.5 FP64 FMA per CUDA Core, 1:2 of FP32)
#                FP16: 1545e6*528*(8*4*16)*2/1e12 = 835.33824 TFLOPS (456 Tensor Cores contracting 8x4x16 FP16 FMAs/clock cycle @ 0% sparsity)
#                TF32: 1545e6*528*(8*4*8)*2/1e12  = 417.66912 TFLOPS (456 Tensor Cores contracting 8x4x8  TF32 FMAs/clock cycle @ 0% sparsity, 1:2 of FP16)
#                (Chiffres déclarés par NVIDIA: 60TF FP32, 30TF FP64, 835TF FP16 non-sparse, 3341TF FP8 sparse.)
#                (Il n'y a aucun agrément sur les specs exacts du NVL sur Internet. C'est ludique.)
#

RAWDATA = {            #             TFLOPS          TFLOPS          TFLOPS           TFLOPS         GB         GB/s        W
    'P100-PCIe-12GB':  dict(fp16= 18.679808, fp32= 9.339904, fp64= 4.669952, tf32=      None, memgb= 12,  membw= 549, tdp=250),
    'P100-PCIe-16GB':  dict(fp16= 18.679808, fp32= 9.339904, fp64= 4.669952, tf32=      None, memgb= 16,  membw= 732, tdp=250),
    'P100-SXM2-16GB':  dict(fp16= 21.217280, fp32=10.608640, fp64= 5.304320, tf32=      None, memgb= 16,  membw= 732, tdp=300),
    'V100-PCIe-16GB':  dict(fp16=112.230400, fp32=14.028800, fp64= 7.014400, tf32=      None, memgb= 16,  membw= 900, tdp=250),
    'V100-PCIe-32GB':  dict(fp16=112.230400, fp32=14.028800, fp64= 7.014400, tf32=      None, memgb= 32,  membw= 900, tdp=250),
    'V100-SXM2-16GB':  dict(fp16=125.337600, fp32=15.667200, fp64= 7.833600, tf32=      None, memgb= 16,  membw= 900, tdp=300),
    'V100-SXM2-32GB':  dict(fp16=125.337600, fp32=15.667200, fp64= 7.833600, tf32=      None, memgb= 32,  membw= 900, tdp=300),
    'V100S-PCIe-32GB': dict(fp16=130.826240, fp32=16.353280, fp64= 8.176640, tf32=      None, memgb= 32,  membw=1134, tdp=250),
    'T4':              dict(fp16= 65.126400, fp32= 8.140800, fp64= 0.254400, tf32=      None, memgb= 16,  membw= 300, tdp= 70),
    'RTX8000':         dict(fp16=130.498560, fp32=16.312320, fp64= 0.509760, tf32=      None, memgb= 48,  membw= 672, tdp=260),
    'A100-PCIe-40GB':  dict(fp16=311.869440, fp32=19.491840, fp64= 9.745920, tf32=155.934720, memgb= 40,  membw=1555, tdp=250),
    'A100-PCIe-80GB':  dict(fp16=311.869440, fp32=19.491840, fp64= 9.745920, tf32=155.934720, memgb= 80,  membw=1555, tdp=250),
    'A100-SXM4-40GB':  dict(fp16=311.869440, fp32=19.491840, fp64= 9.745920, tf32=155.934720, memgb= 40,  membw=1555, tdp=400),
    'A100-SXM4-80GB':  dict(fp16=311.869440, fp32=19.491840, fp64= 9.745920, tf32=155.934720, memgb= 80,  membw=1555, tdp=400),
    'A6000':           dict(fp16=154.828800, fp32=38.707200, fp64= 0.604800, tf32= 77.414400, memgb= 48,  membw= 768, tdp=300),
    'L40S':            dict(fp16=366.428160, fp32=91.607040, fp64= 1.431360, tf32=183.214080, memgb= 48,  membw= 864, tdp=350),
    'H100-PCIe-80GB':  dict(fp16=756.449280, fp32=51.217920, fp64=25.608960, tf32=378.224640, memgb= 80,  membw=2039, tdp=350),
    'H100-SXM5-80GB':  dict(fp16=989.429760, fp32=66.908160, fp64=33.454080, tf32=494.714880, memgb= 80,  membw=3352, tdp=700),
    'H100-NVL-94GB':   dict(fp16=835.500000, fp32=60.000000, fp64=30.000000, tf32=417.250000, memgb= 94,  membw=3938, tdp=400),
}


@fom
def fom_ugr(name, **kwargs):
    data = RAWDATA[name]
    ref  = RAWDATA['A100-SXM4-40GB']
    return 1.6 * (data['fp16']  / ref['fp16']) + \
           1.6 * (data['fp32']  / ref['fp32']) + \
           0.8 * (data['memgb'] / ref['memgb'])

@fom
def fom_iguane(name, *, args=None):
    data = RAWDATA[name].copy()
    ref  = RAWDATA['A100-SXM4-80GB']
    data['tf32'] = data['tf32'] or data['fp32']
    weight_fp16  = 0.2
    weight_fp32  = 0.2
    weight_tf32  = 0.2
    weight_memgb = 0.2
    weight_membw = 0.2
    return weight_fp16  * (data['fp16']  / ref['fp16'])  + \
           weight_fp32  * (data['fp32']  / ref['fp32'])  + \
           weight_tf32  * (data['tf32']  / ref['tf32'])  + \
           weight_memgb * (data['memgb'] / ref['memgb']) + \
           weight_membw * (data['membw'] / ref['membw'])


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--reverse',   '-r', action='store_true',
                      help="Reverse listing")
    argp.add_argument('--sort',      '-s', action='store_true',
                      help="Sort GPU listing by unit")
    argp.add_argument('--list-units',      action='store_true',
                      help="List known Units")
    argp.add_argument('--list-gpus', '-l', action='store_true',
                      help="List known GPUs")
    argp.add_argument('--gpu',  '-G', type=str, default=None,
                      help="Selected GPU")
    
    ogrp = argp.add_argument_group('OUTPUT FORMAT', 'Output format control')
    ogrp.add_argument('--delimiter', '-d', type=str, default=',',
                      help="Delimiter for --parsable output")
    omtx = ogrp.add_mutually_exclusive_group()
    omtx.add_argument('--json',      '-j', action='store_true',
                      help="Dump output as JSON")
    omtx.add_argument('--parsable',  '-p', action='store_true',
                      help="Dump output as delimited text")
    
    ugrp = argp.add_argument_group('UNITS', 'Unit/FoM selection')
    umtx = ugrp.add_mutually_exclusive_group()
    umtx.add_argument('--unit', '-u', '--fom', type=str.lower, default='iguane',
                      choices={'iguane', 'iguana', 'rgu', 'ugr'},
                      help="Select unit-equivalence (FoM) to compute for every GPU")
    umtx.add_argument('--iguane', action='store_const', dest='unit', const='iguane',
                      help="Select IGUANE/IGUANA unit-equivalence")
    umtx.add_argument('--iguana', action='store_const', dest='unit', const='iguane',
                      help="Select IGUANE/IGUANA unit-equivalence")
    umtx.add_argument('--ugr',    action='store_const', dest='unit', const='ugr',
                      help="Select UGR/RGU unit-equivalence")
    umtx.add_argument('--rgu',    action='store_const', dest='unit', const='ugr',
                      help="Select UGR/RGU unit-equivalence")
    
    # Parse Arguments
    args = argp.parse_args(sys.argv[1:])
    args.unit = {'iguana': 'iguane', 'rgu': 'ugr'}.get(args.unit, args.unit)
    assert args.delimiter, "Delimiter cannot be empty string!"
    
    
    if   args.list_units:
        units = ['IGUANE', 'UGR']
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
