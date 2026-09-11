import httpx
from typing import List, Dict

VERCEL_STANDARD_LIMIT_MB = 250  # uncompressed, standard path
VERCEL_FLUID_LIMIT_MB = 500     # uncompressed, Python w/ Fluid Compute

HEAVY_PACKAGE_HINTS = {
    "numpy": "Often pulled in transitively — confirm you actually need it directly.",
    "pandas": "Consider 'polars' for a much smaller, faster alternative.",
    "torch": "PyTorch is huge (~500MB+). Consider 'onnxruntime' for inference-only, or run ML in a separate service.",
    "tensorflow": "Very large. Consider 'tensorflow-cpu' or 'tflite-runtime' for inference-only use.",
    "scipy": "Large binary dependency — check if numpy alone covers your need.",
    "opencv-python": "Use 'opencv-python-headless' instead — drops GUI deps, much smaller.",
    "transformers": "Very large with optional deps — 'transformers[torch]' silently pulls in torch too.",
    "langchain": "Full 'langchain' pulls many integrations. Use 'langchain-core' + only what you need.",
    "qdrant-client": "Check if a lighter HTTP-only client covers your use case (skip gRPC extras).",
    "matplotlib": "If used for one chart type, consider a lighter plotting lib.",
}

def parse_requirements(text: str) -> List[str]:
    packages = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split(";")[0].strip()
        packages.append(line)
    return packages

def split_name_version(requirement: str):
    for sep in ["==", ">=", "<=", "~=", ">", "<"]:
        if sep in requirement:
            name, version = requirement.split(sep, 1)
            return name.strip(), version.strip()
    return requirement.strip(), None

async def get_package_size(client: httpx.AsyncClient, name: str, version: str = None) -> Dict:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"name": name, "version": version, "size_bytes": 0, "error": str(e)}

    releases = data.get("releases", {})
    target_version = version or data.get("info", {}).get("version")
    files = releases.get(target_version, []) or data.get("urls", [])

    wheel_files = [f for f in files if f.get("packagetype") == "bdist_wheel"]
    chosen = wheel_files[0] if wheel_files else (files[0] if files else None)
    size_bytes = chosen.get("size", 0) if chosen else 0

    return {
        "name": name,
        "version": target_version,
        "size_bytes": size_bytes,
        "error": None if chosen else "No release files found",
    }

async def analyze_requirements(text: str) -> Dict:
    requirements = parse_requirements(text)
    results = []

    async with httpx.AsyncClient() as client:
        for req in requirements:
            name, version = split_name_version(req)
            info = await get_package_size(client, name, version)
            results.append(info)

    total_bytes = sum(r["size_bytes"] for r in results)
    estimated_uncompressed_mb = (total_bytes * 3) / (1024 * 1024)  # rule-of-thumb expansion
    total_compressed_mb = total_bytes / (1024 * 1024)

    results_sorted = sorted(results, key=lambda r: r["size_bytes"], reverse=True)

    warnings = []
    for r in results_sorted:
        hint = HEAVY_PACKAGE_HINTS.get(r["name"].lower())
        if hint:
            warnings.append({"package": r["name"], "hint": hint})

    verdict = "PASS — well within limits"
    if estimated_uncompressed_mb > VERCEL_FLUID_LIMIT_MB:
        verdict = "FAIL — exceeds 500MB Fluid Compute limit"
    elif estimated_uncompressed_mb > VERCEL_STANDARD_LIMIT_MB:
        verdict = "WARN — exceeds 250MB standard limit, requires Fluid Compute opt-in"

    return {
        "packages": results_sorted,
        "total_compressed_mb": round(total_compressed_mb, 2),
        "estimated_uncompressed_mb": round(estimated_uncompressed_mb, 2),
        "verdict": verdict,
        "warnings": warnings,
    }
