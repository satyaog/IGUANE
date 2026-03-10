# IGUANE: A tool to explore Figures-of-Merit for some GPUs

Compute and compare GPU Figure-of-Merit scores — including DRAC's [RGU/UGR](https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Reference_GPU_Units) and a custom variant called IGUANE/IGUANA.

---

## What is a Figure-of-Merit?

A Figure-of-Merit (FoM) is a single score that summarizes a GPU's usefulness for ML workloads. Rather than forcing you to compare six raw specs at once, a FoM combines them into a weighted sum normalized against a reference GPU. A score of `2.0` means "twice as useful as the reference GPU" according to that weighting.

**Raw specs used:**

| Field    | Description |
|----------|-------------|
| `fp16`   | Mixed-precision tensor throughput (Tensor Cores), in TFLOPS. What most deep learning training uses. Falls back to fp32 on older GPUs without Tensor Core fp16 support. |
| `fp32`   | Scalar single-precision throughput, in TFLOPS. Classical compute fallback. |
| `tf32`   | TensorFloat-32 throughput (NVIDIA Ampere+), in TFLOPS. Faster than fp32 but less precise than fp16. Falls back to fp32 on older GPUs. |
| `memgb`  | GPU memory capacity, in GB. Limits max model size and batch size. |
| `membw`  | Memory bandwidth, in GB/s. Limits how fast data flows to/from compute cores; critical for inference workloads. |

**Formula:**

```
FoM = Σ (weight_i × spec_i / ref_spec_i)
```

Each spec is divided by the reference GPU's value, so the reference GPU always scores 1.0 on that factor (before weighting).

**Worked example — RGU for H100-SXM5-80GB:**

RGU uses weights `fp16×1.6 + fp32×1.6 + memgb×0.8` relative to the A100-SXM4-40GB.

| Factor  | Weight | GPU value  | Ref value  | Ratio  | Contribution |
|---------|--------|------------|------------|--------|--------------|
| fp16    |  1.6   | 989.43 T   | 311.87 T   | 3.173  | 5.077        |
| fp32    |  1.6   |  66.91 T   |  19.49 T   | 3.433  | 5.492        |
| memgb   |  0.8   |  80 GB     |  40 GB     | 2.000  | 1.600        |
| **RGU** |        |            |            |        | **12.17**    |

---

## FoM versions

### RGU / 1.0 (default)

DRAC's official metric used for Alliance Canada cluster allocation requests. Uses three factors — `fp16`, `fp32`, and `memgb` — relative to the A100-SXM4-40GB. Captures mixed-precision compute throughput and memory capacity, which are the dominant constraints for most training workloads.

Weights: `fp16×1.6 + fp32×1.6 + memgb×0.8`

Use when: submitting DRAC allocation requests or comparing against Alliance Canada's published RGU values.

### 2.0 (experimental)

A five-factor version that adds `tf32` (tensor compute) and `membw` (memory bandwidth) to the RGU factors. Memory-related factors (`memgb` + `membw`) together account for half the total weight, making this version more sensitive to bandwidth-bound workloads. Reference GPU: A100-SXM4-80GB.

Weights: `fp16×0.2 + fp32×0.1 + tf32×0.2 + memgb×0.25 + membw×0.25`

Use when: evaluating GPUs for large-batch inference or LLM serving, where memory bandwidth is the primary bottleneck.

### IGUANE (equal-weight)

A five-factor version that treats all dimensions equally — compute (fp16, fp32, tf32) and memory (memgb, membw) each get weight 0.2. Provides a balanced, unbiased comparison. Reference GPU: A100-SXM4-80GB.

Weights: `fp16×0.2 + fp32×0.2 + tf32×0.2 + memgb×0.2 + membw×0.2`

Use when: you want a general-purpose comparison that doesn't favour any single hardware dimension.

### Decision table

| Situation                                    | Use                  |
|----------------------------------------------|----------------------|
| DRAC cluster allocation request              | RGU (1.0)            |
| Comparing Alliance Canada cluster GPU fleets | RGU (1.0)            |
| Evaluating GPUs for LLM inference / serving  | 2.0                  |
| General balanced GPU comparison              | IGUANE               |
| Custom workload with specific bottleneck     | `--custom-weights`   |

---

## Quickstart

**List all GPUs sorted by RGU score:**

    $ python -m iguane -sr
    25.24 MI325X
    23.96 MI300X
    14.84 RTX-PRO-6000-Blackwell-Workstation-Edition
    13.39 H200-SXM5-141GB
    12.17 H100-SXM5-80GB
    12.03 H200-NVL-141GB
    11.09 H100-NVL-94GB
    10.36 L40S
    10.31 RTX-5090
     9.69 H100-PCIe-80GB
     ...

**Compute total RGU for a cluster inventory JSON:**

The `-i` flag reads a JSON file mapping GPU names to counts and outputs the total FoM-weighted GPU equivalents for the whole cluster. The simplest form gives a single number — the total RGU budget of the cluster:

    $ python -m iguane -i mila.json
    5886.69

Add `-sj` to get a full JSON breakdown sorted by each GPU type's contribution, which is useful for understanding which GPUs dominate the cluster's RGU budget:

    $ python -m iguane -sji mila.json
    {
      "breakdown": {
        "A6000": 39.453049645390074,
        "A100-SXM4-40GB": 128.0,
        "V100-SXM2-32GB": 143.86836879432624,
        "H100-SXM5-80GB": 194.6931442080378,
        "A100-SXM4-80GB": 422.4000000000001,
        "RTX8000": 1187.4042553191491,
        "L40S": 3770.8678959810877
      },
      "total": 5886.686713947991
    }

**Get raw RGU counts as JSON (sorted descending):**

The `-j` flag emits JSON instead of the human-readable table. Combined with `-s -r` (sort, reverse) and `-u rgu`, this outputs each GPU's absolute RGU count — i.e. how many Reference GPU Units it is worth. Because RGU weights sum to 4.0 (fp16×1.6 + fp32×1.6 + memgb×0.8), the reference GPU A100-SXM4-40GB scores exactly 4.0 RGU. This is the format most directly comparable to DRAC's published allocation tables:

    $ python -m iguane -jsru rgu
    {
      "H100-SXM5-80GB": 12.168321513002363,
      "H100-NVL-94GB": 11.091546985815604,
      "L40S": 10.35952718676123,
      "H100-PCIe-80GB": 9.685106382978724,
      "A6000": 4.931631205673759,
      "A100-SXM4-80GB": 4.800000000000001,
      "A100-PCIe-80GB": 4.800000000000001,
      "A100-SXM4-40GB": 4.0,
      "A100-PCIe-40GB": 4.0,
      "RTX8000": 2.9685106382978725,
      "V100S-PCIe-32GB": 2.6535539795114267,
      "V100-SXM2-32GB": 2.5690780141843974,
      "V100-PCIe-32GB": 2.367344365642238,
      "V100-SXM2-16GB": 2.2490780141843976,
      "V100-PCIe-16GB": 2.047344365642238,
      "T4": 1.3223640661938536,
      "P100-SXM2-16GB": 1.2996690307328604,
      "P100-PCIe-16GB": 1.182505910165485,
      "P100-PCIe-12GB": 1.1025059101654848
    }

**Output in a custom parsable format (useful for scripting):**

The `-p` flag switches to delimiter-separated output instead of the default human-readable table. Use `-d` to set the delimiter — here `": "` produces `GPU: value` lines that are easy to parse with `awk`, `cut`, or any line-oriented tool:

    $ python -m iguane --iguane -pd": "
    P100-PCIe-12GB: 0.22040341999886826
    P100-PCIe-16GB: 0.2539403974908297
    P100-SXM2-16GB: 0.2702130531251874
    V100-PCIe-16GB: 0.389666843190366
    V100-PCIe-32GB: 0.429666843190366
    V100-SXM2-16GB: 0.4169849414304501
    V100-SXM2-32GB: 0.4569849414304501
    V100S-PCIe-32GB: 0.49852085809099334
    T4: 0.2143220096336331
    RTX8000: 0.4784166837700395
    A100-PCIe-40GB: 0.9000000000000001
    A100-PCIe-80GB: 1.0
    A100-SXM4-40GB: 0.9000000000000001
    A100-SXM4-80GB: 1.0
    A6000: 0.8145228158992954
    L40S: 1.641054479943445
    H100-PCIe-80GB: 1.9579954847095848
    H100-SXM5-80GB: 2.586680957484816
    H100-NVL-94GB: 2.4280983913222136

The inventory JSON maps GPU names to counts, e.g.:

```json
{"A100-SXM4-80GB": 32, "H100-SXM5-80GB": 16}
```

---

## CLI Reference

### General

| Flag                   | Short | Description                                                                                         |
|------------------------|-------|-----------------------------------------------------------------------------------------------------|
| `--reverse`            | `-r`  | Reverse the GPU listing order                                                                       |
| `--sort`               | `-s`  | Sort GPU listing by FoM value                                                                       |
| `--list-gpus`          | `-l`  | Print all known GPU names                                                                           |
| `--list-units`         |       | Print all known unit/FoM names                                                                      |
| `--list-fom-versions`  |       | Print all known FoM ponderation versions                                                            |
| `--dump-raw`           |       | Dump raw GPU spec data as JSON                                                                      |
| `--input PATH`         | `-i`  | Path to a cluster inventory JSON (maps GPU names to counts); outputs total FoM-weighted equivalents |
| `--gpu PATTERN`        | `-G`  | Filter output to GPUs matching a name prefix or glob pattern                                        |
| `--verbose`            | `-v`  | Increase verbosity (repeat for more: `-vv`, `-vvv`)                                                 |

### Output format

| Flag              | Short | Description                                     |
|-------------------|-------|-------------------------------------------------|
| `--json`          | `-j`  | Output as JSON                                  |
| `--parsable`      | `-p`  | Output as delimiter-separated text              |
| `--delimiter STR` | `-d`  | Delimiter used with `--parsable` (default: `,`) |

### Units / FoM selection

| Flag                          | Short | Description                                                                         |
|-------------------------------|-------|-------------------------------------------------------------------------------------|
| `--unit NAME` / `--fom NAME`  | `-u`  | Select the FoM to compute (`ugr`, `iguane`, or any registered name; default: `ugr`) |
| `--ugr` / `--rgu`             |       | Shorthand to select UGR/RGU (DRAC's Reference GPU Unit)                             |
| `--iguane` / `--iguana`       |       | Shorthand to select IGUANE/IGUANA FoM                                               |
| `--fom-version VER`           |       | Select FoM ponderation version (`1.0`, `2.0`, `iguane`, …)                          |
| `--norm`                      |       | Normalize FoM weights to sum to 1.0                                                 |
| `--custom-weights JSON`       |       | Override weights with a JSON object: `{"ref": "GPU-NAME", "fp16": 0.0, …}`          |
