import json
import pathlib
from types import SimpleNamespace

from mcp.server.fastmcp import FastMCP

from iguane.fom import RAWDATA, FOM_VERSIONS, _CURRENT_FOM_VERSION, fom_ugr

mcp = FastMCP("iguane")

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

_FOM_VERSION_DESCRIPTIONS = {
    "1.0": (
        "Equivalent to DRAC's official RGU/UGR metric used for Alliance cluster allocations. "
        "Weights: fp16×1.6 + fp32×1.6 + memgb×0.8, normalized relative to A100-SXM4-40GB. "
        "Captures mixed-precision compute and memory capacity. "
        "Use this if you need to compare against DRAC's published RGU values."
    ),
    "ugr": "Alias for version 1.0 (DRAC's RGU/UGR metric).",
    "2.0-0": (
        "Experimental predecessor to 2.0. Same weights as 2.0, kept for historical reference."
    ),
    "2.0": (
        "Experimental 5-factor version. "
        "Weights: fp16×0.2 + fp32×0.1 + tf32×0.2 + memgb×0.25 + membw×0.25, normalized relative to A100-SXM4-80GB. "
        "Adds memory bandwidth and TF32 tensor performance on top of the 1.0 factors. "
        "Weights memory-related factors (0.5 total) more heavily than compute (0.5 total, split between fp16/fp32/tf32). "
        "Use this if memory bandwidth is critical to your workloads (e.g. large-batch inference, LLM serving)."
    ),
    "iguane": (
        "IGUANE's custom equal-weight 5-factor version. "
        "Weights: fp16×0.2 + fp32×0.2 + tf32×0.2 + memgb×0.2 + membw×0.2, normalized relative to A100-SXM4-80GB. "
        "Treats all five factors equally: mixed-precision compute, scalar compute, tensor compute, memory size, and memory bandwidth. "
        "Use this for a balanced view that doesn't over-emphasize any single dimension."
    ),
}


def _make_args(fom_version: str) -> SimpleNamespace:
    return SimpleNamespace(fom_version=fom_version, custom_weights=None, norm=False)


@mcp.tool()
def list_gpus() -> str:
    """List all available GPU names in the database.

    Returns:
        str: A newline-separated list of GPU names.
    """
    return "\n".join(sorted(RAWDATA.keys()))


@mcp.tool()
def get_gpu_specs(gpu_name: str) -> str:
    """Get the raw specifications for a GPU.

    Args:
        gpu_name (str): The exact GPU name (e.g. "A100-SXM4-80GB").

    Returns:
        str: JSON object with fields fp16, fp32, fp64, tf32 (TFLOPS),
             memgb (GB), membw (GB/s), tdp (W), reldate.
    """
    if gpu_name not in RAWDATA:
        available = ", ".join(sorted(RAWDATA.keys()))
        return f"Unknown GPU '{gpu_name}'. Available GPUs: {available}"
    return json.dumps(RAWDATA[gpu_name], indent=2)


@mcp.tool()
def describe_fom_versions() -> str:
    """Describe all available Figure-of-Merit (FoM) versions with human-readable explanations.

    Returns:
        str: JSON object mapping each version name to its description, weights, and reference GPU.
    """
    result = {}
    for name, weights in FOM_VERSIONS.items():
        result[name] = {
            "description": _FOM_VERSION_DESCRIPTIONS.get(name, "No description available."),
            "reference_gpu": weights.get("ref", "A100-SXM4-40GB"),
            "weights": {k: v for k, v in weights.items() if k != "ref"},
        }
    return json.dumps(result, indent=2)


@mcp.tool()
def compute_fom(gpu_name: str, fom_version: str = _CURRENT_FOM_VERSION) -> str:
    """Compute the Figure-of-Merit (FoM) for a single GPU.

    The default version (1.0 / ugr) is equivalent to DRAC's RGU metric,
    normalized relative to the A100-SXM4-40GB (which scores 1.0).

    Args:
        gpu_name (str): The exact GPU name (e.g. "H100-SXM5-80GB").
        fom_version (str): FoM version to use. One of: "1.0" (ugr/RGU),
            "2.0", "iguane". Defaults to "1.0".

    Returns:
        str: The FoM value as a float.
    """
    if gpu_name not in RAWDATA:
        available = ", ".join(sorted(RAWDATA.keys()))
        return f"Unknown GPU '{gpu_name}'. Available GPUs: {available}"
    if fom_version not in FOM_VERSIONS:
        available = ", ".join(FOM_VERSIONS.keys())
        return f"Unknown FoM version '{fom_version}'. Available: {available}"
    return str(fom_ugr(gpu_name, args=_make_args(fom_version)))


@mcp.tool()
def compute_cluster_fom(inventory: dict[str, int], fom_version: str = _CURRENT_FOM_VERSION) -> str:
    """Compute the total FoM-weighted GPU equivalents for a cluster inventory.

    Args:
        inventory (dict[str, int]): Mapping of GPU name to count
            (e.g. {"A100-SXM4-80GB": 64, "H100-SXM5-80GB": 16}).
        fom_version (str): FoM version to use. One of: "1.0" (ugr/RGU),
            "2.0", "iguane". Defaults to "1.0".

    Returns:
        str: JSON object with per-GPU FoM contributions and the grand total.
    """
    if fom_version not in FOM_VERSIONS:
        available = ", ".join(FOM_VERSIONS.keys())
        return f"Unknown FoM version '{fom_version}'. Available: {available}"

    args = _make_args(fom_version)
    results = {}
    total = 0.0
    errors = []

    for gpu_name, count in inventory.items():
        if gpu_name not in RAWDATA:
            errors.append(f"Unknown GPU '{gpu_name}'")
            continue
        fom_value = fom_ugr(gpu_name, args=args)
        contribution = fom_value * count
        results[gpu_name] = {"count": count, "fom_per_gpu": fom_value, "total_fom": contribution}
        total += contribution

    output = {"fom_version": fom_version, "gpus": results, "grand_total_fom": total}
    if errors:
        output["errors"] = errors
    return json.dumps(output, indent=2)


@mcp.tool()
def list_mila_inventories() -> str:
    """List available Mila cluster inventory snapshots bundled with IGUANE.

    Each snapshot represents the Mila cluster GPU fleet at a given period.
    Period names follow the format YYYY_SN (e.g. "2025_S2" = year 2025, semester 2).

    Returns:
        str: Newline-separated list of available period names.
    """
    mila_dir = _DATA_DIR / "mila"
    periods = sorted(p.stem for p in mila_dir.glob("*.json"))
    return "\n".join(periods)


@mcp.tool()
def compute_mila_cluster_fom(period: str = "2025_S2", fom_version: str = _CURRENT_FOM_VERSION) -> str:
    """Compute the total FoM for the Mila cluster at a given period.

    Args:
        period (str): Snapshot period in YYYY_SN format (e.g. "2025_S2").
            Use list_mila_inventories() to see available periods.
        fom_version (str): FoM version to use. One of: "1.0" (ugr/RGU),
            "2.0", "iguane". Defaults to "1.0".

    Returns:
        str: JSON object with per-GPU FoM contributions and the grand total.
    """
    mila_dir = _DATA_DIR / "mila"
    path = mila_dir / f"{period}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in mila_dir.glob("*.json")))
        return f"Unknown period '{period}'. Available: {available}"

    with open(path) as f:
        inventory = json.load(f)

    # Delegate to compute_cluster_fom logic
    if fom_version not in FOM_VERSIONS:
        available = ", ".join(FOM_VERSIONS.keys())
        return f"Unknown FoM version '{fom_version}'. Available: {available}"

    args = _make_args(fom_version)
    results = {}
    total = 0.0
    errors = []

    for gpu_name, count in inventory.items():
        if gpu_name not in RAWDATA:
            errors.append(f"Unknown GPU '{gpu_name}' (skipped)")
            continue
        fom_value = fom_ugr(gpu_name, args=args)
        contribution = fom_value * count
        results[gpu_name] = {"count": count, "fom_per_gpu": fom_value, "total_fom": contribution}
        total += contribution

    output = {"period": period, "fom_version": fom_version, "gpus": results, "grand_total_fom": total}
    if errors:
        output["errors"] = errors
    return json.dumps(output, indent=2)


@mcp.tool()
def gpu_equivalents(total_fom: float, target_gpu: str, fom_version: str = _CURRENT_FOM_VERSION) -> str:
    """Given a total FoM score, compute how many GPUs of a target type that represents.

    Useful for answering: "Our cluster has X RGU — how many H100s is that equivalent to?"

    Args:
        total_fom (float): Total FoM score (e.g. grand_total_fom from compute_mila_cluster_fom).
        target_gpu (str): The GPU to express the equivalent in (e.g. "H100-SXM5-80GB").
        fom_version (str): FoM version to use. Must match the version used to compute total_fom.

    Returns:
        str: JSON object with the equivalent count and per-GPU FoM of the target GPU.
    """
    if target_gpu not in RAWDATA:
        available = ", ".join(sorted(RAWDATA.keys()))
        return f"Unknown GPU '{target_gpu}'. Available GPUs: {available}"
    if fom_version not in FOM_VERSIONS:
        available = ", ".join(FOM_VERSIONS.keys())
        return f"Unknown FoM version '{fom_version}'. Available: {available}"

    fom_per_gpu = fom_ugr(target_gpu, args=_make_args(fom_version))
    equivalent_count = total_fom / fom_per_gpu

    return json.dumps({
        "total_fom": total_fom,
        "target_gpu": target_gpu,
        "fom_per_gpu": fom_per_gpu,
        "equivalent_count": equivalent_count,
        "fom_version": fom_version,
    }, indent=2)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
