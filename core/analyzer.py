import httpx
import re
from typing import List, Dict, Set, Optional

VERCEL_STANDARD_LIMIT_MB = 250
VERCEL_FLUID_LIMIT_MB = 500
MAX_DEPTH = 3
ROUTE_SPLIT_THRESHOLD_MB = 15

ALTERNATIVES = {
    "pandas": "polars",
    "torch": "onnxruntime",
    "tensorflow": "tflite-runtime",
    "opencv-python": "opencv-python-headless",
    "scipy": None,
}

GPU_TAGS = ("cu11", "cu12", "cu13", "rocm", "cuda")
NVIDIA_CUDA_PREFIXES = ("nvidia-", "triton", "cuda-", "cusparselt", "nvshmem")

NATIVE_INIT_PACKAGES = {
    "torch", "tensorflow", "numpy", "scipy", "opencv-python",
    "pandas", "transformers", "onnxruntime", "scikit-learn",
}


def parse_requirements(text: str) -> List[str]:
    packages = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        packages.append(line.split(";")[0].strip())
    return packages


def split_name_version(requirement: str):
    for sep in ["==", ">=", "<=", "~=", ">", "<"]:
        if sep in requirement:
            name, version = requirement.split(sep, 1)
            return name.strip().lower(), version.strip()
    return requirement.strip().lower(), None


def extract_dep_name(requires_dist_entry: str) -> Optional[str]:
    if "extra ==" in requires_dist_entry:
        return None
    match = re.match(r"^([A-Za-z0-9._-]+)", requires_dist_entry.strip())
    return match.group(1).lower() if match else None


async def fetch_package_json(client: httpx.AsyncClient, name: str):
    try:
        resp = await client.get(f"https://pypi.org/pypi/{name}/json", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def pick_best_size(data: dict, version: str = None) -> int:
    releases = data.get("releases", {})
    target = version or data.get("info", {}).get("version")
    files = releases.get(target, []) or data.get("urls", [])

    wheels = [f for f in files if f.get("packagetype") == "bdist_wheel"]
    if not wheels:
        non_wheels = [f for f in files if f.get("packagetype") != "bdist_wheel"]
        return non_wheels[0].get("size", 0) if non_wheels else 0

    pure = [w for w in wheels if "none-any" in w.get("filename", "")]
    if pure:
        return pure[0].get("size", 0)

    linux_cpu = [
        w for w in wheels
        if "manylinux" in w.get("filename", "")
        and not any(tag in w.get("filename", "").lower() for tag in GPU_TAGS)
    ]
    if linux_cpu:
        return min(w.get("size", 0) for w in linux_cpu)

    return min(w.get("size", 0) for w in wheels)


async def resolve_tree(client: httpx.AsyncClient, name: str, version: str,
                        visited: Set[str], depth: int = 0) -> List[Dict]:
    if name in visited or depth > MAX_DEPTH:
        return []
    visited.add(name)

    data = await fetch_package_json(client, name)
    if not data:
        return [{"name": name, "version": version, "size_bytes": 0,
                  "depth": depth, "error": "not found on PyPI"}]

    size = pick_best_size(data, version)
    node = {"name": name, "version": version or data["info"]["version"],
            "size_bytes": size, "depth": depth, "error": None}

    results = [node]
    requires = data.get("info", {}).get("requires_dist") or []
    for entry in requires:
        dep_name = extract_dep_name(entry)
        if dep_name and dep_name not in visited:
            results.extend(await resolve_tree(client, dep_name, None, visited, depth + 1))

    return results


async def suggest_alternative(client: httpx.AsyncClient, name: str):
    alt_name = ALTERNATIVES.get(name)
    if not alt_name:
        return None
    data = await fetch_package_json(client, alt_name)
    if not data:
        return None
    alt_size = pick_best_size(data)
    return {"name": alt_name, "size_bytes": alt_size}


def build_fixed_requirements(original_reqs: List[str], alternatives: List[Dict]) -> str:
    alt_map = {a["original"]: a["alternative"] for a in alternatives}
    fixed_lines = []
    for req in original_reqs:
        name, _ = split_name_version(req)
        fixed_lines.append(alt_map.get(name, req))
    return "\n".join(fixed_lines)


def estimate_cold_start(estimated_uncompressed_mb: float, direct_names: Set[str]) -> Dict:
    native_hits = sorted(direct_names & NATIVE_INIT_PACKAGES)

    if estimated_uncompressed_mb < 50 and not native_hits:
        return {
            "tier": "Fast",
            "note": "Small, pure-Python bundle — cold starts should typically stay under ~300ms.",
        }
    elif estimated_uncompressed_mb < 150 and len(native_hits) <= 1:
        native_note = f" plus native package init ({native_hits[0]})" if native_hits else ""
        return {
            "tier": "Moderate",
            "note": f"Bundle size{native_note} will add noticeable cold start latency on the first request after idle — likely several hundred ms to ~1s.",
        }
    else:
        heavy_list = ", ".join(native_hits) if native_hits else "large bundle size"
        return {
            "tier": "Heavy",
            "note": f"Large bundle with compiled/native packages ({heavy_list}) — expect 1-3s+ cold starts. Fluid Compute keeps instances warmer between requests and helps here; splitting heavy packages into separate functions (see below) reduces the size any single cold start has to load.",
        }


def suggest_route_split(unique_nodes: List[Dict]) -> Optional[Dict]:
    direct = [n for n in unique_nodes if n["depth"] == 0]
    heavy_direct = sorted(
        [n for n in direct if n["size_bytes"] / 1024 / 1024 > ROUTE_SPLIT_THRESHOLD_MB],
        key=lambda n: n["size_bytes"], reverse=True,
    )

    if len(heavy_direct) < 2:
        return None

    heavy_names = {n["name"] for n in heavy_direct}
    lightweight = [n["name"] for n in direct if n["name"] not in heavy_names]
    lightweight_size_mb = sum(
        n["size_bytes"] for n in direct if n["name"] in lightweight
    ) / 1024 / 1024

    suggestions = []
    for pkg in heavy_direct:
        suggestions.append({
            "function_name": f"api/{pkg['name'].replace('-', '_')}_handler.py",
            "packages": [pkg["name"]] + lightweight,
            "size_mb": round(pkg["size_bytes"] / 1024 / 1024 + lightweight_size_mb, 2),
        })

    return {
        "heavy_count": len(heavy_direct),
        "shared_base": lightweight,
        "suggestions": suggestions,
        "note": (
            f"You have {len(heavy_direct)} large, independent packages bundled into a single "
            "function. Vercel bundles each Python function based on what that file actually "
            "imports — so isolating each heavy package into its own route file means routes "
            "that don't use it stay small and fast, instead of every route paying for the "
            "heaviest one."
        ),
    }


def detect_cuda_bloat(unique_nodes: List[Dict], top_level_names: Set[str]) -> Optional[Dict]:
    if "torch" not in top_level_names:
        return None

    cuda_packages = [
        n for n in unique_nodes
        if any(n["name"].startswith(p) for p in NVIDIA_CUDA_PREFIXES)
    ]
    if not cuda_packages:
        return None

    cuda_total_bytes = sum(n["size_bytes"] for n in cuda_packages)
    cuda_total_mb = cuda_total_bytes / 1024 / 1024

    return {
        "cuda_total_mb": round(cuda_total_mb, 2),
        "cuda_total_bytes": cuda_total_bytes,
        "package_names": [n["name"] for n in cuda_packages],
        "package_count": len(cuda_packages),
        "fix": "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "note": (
            f"{len(cuda_packages)} CUDA/GPU packages ({round(cuda_total_mb, 2)} MB) got pulled in "
            "because PyPI's default torch build assumes a GPU target. Vercel Functions are CPU-only, "
            "so none of this is needed — installing from PyTorch's dedicated CPU-only index avoids it entirely."
        ),
    }


def build_corrected_estimate(total_bytes: int, cuda_bloat: Optional[Dict]) -> Optional[Dict]:
    if not cuda_bloat:
        return None

    corrected_bytes = total_bytes - cuda_bloat["cuda_total_bytes"]
    corrected_uncompressed_mb = (corrected_bytes * 3) / (1024 * 1024)

    if corrected_uncompressed_mb > VERCEL_FLUID_LIMIT_MB:
        corrected_verdict = "FAIL — exceeds 500MB Fluid Compute limit"
    elif corrected_uncompressed_mb > VERCEL_STANDARD_LIMIT_MB:
        corrected_verdict = "WARN — exceeds 250MB standard limit, needs Fluid Compute opt-in"
    else:
        corrected_verdict = "PASS — within limits"

    corrected_pct = min(100, round((corrected_uncompressed_mb / VERCEL_FLUID_LIMIT_MB) * 100, 1))

    return {
        "estimated_uncompressed_mb": round(corrected_uncompressed_mb, 2),
        "verdict": corrected_verdict,
        "pct_of_fluid_limit": corrected_pct,
    }


async def analyze_requirements(text: str) -> Dict:
    requirements = parse_requirements(text)
    visited: Set[str] = set()
    all_nodes: List[Dict] = []

    async with httpx.AsyncClient() as client:
        for req in requirements:
            name, version = split_name_version(req)
            all_nodes.extend(await resolve_tree(client, name, version, visited))

        seen = {}
        for node in all_nodes:
            if node["name"] not in seen:
                seen[node["name"]] = node
        unique_nodes = list(seen.values())

        total_bytes = sum(n["size_bytes"] for n in unique_nodes)
        estimated_uncompressed_mb = (total_bytes * 3) / (1024 * 1024)
        total_compressed_mb = total_bytes / (1024 * 1024)

        top_level_names = {split_name_version(r)[0] for r in requirements}
        alternatives = []
        for name in top_level_names:
            alt = await suggest_alternative(client, name)
            if alt:
                original = seen.get(name)
                if original:
                    savings_mb = (original["size_bytes"] - alt["size_bytes"]) / (1024 * 1024)
                    alternatives.append({
                        "original": name,
                        "original_mb": round(original["size_bytes"] / 1024 / 1024, 2),
                        "alternative": alt["name"],
                        "alternative_mb": round(alt["size_bytes"] / 1024 / 1024, 2),
                        "savings_mb": round(savings_mb, 2),
                    })

    unique_nodes.sort(key=lambda n: n["size_bytes"], reverse=True)

    if estimated_uncompressed_mb > VERCEL_FLUID_LIMIT_MB:
        verdict = "FAIL — exceeds 500MB Fluid Compute limit"
    elif estimated_uncompressed_mb > VERCEL_STANDARD_LIMIT_MB:
        verdict = "WARN — exceeds 250MB standard limit, needs Fluid Compute opt-in"
    else:
        verdict = "PASS — within limits"

    pct_of_fluid_limit = min(100, round((estimated_uncompressed_mb / VERCEL_FLUID_LIMIT_MB) * 100, 1))
    fixed_requirements = build_fixed_requirements(requirements, alternatives) if alternatives else None

    cold_start = estimate_cold_start(estimated_uncompressed_mb, top_level_names)
    route_split = suggest_route_split(unique_nodes)
    cuda_bloat = detect_cuda_bloat(unique_nodes, top_level_names)
    corrected_estimate = build_corrected_estimate(total_bytes, cuda_bloat)

    return {
        "packages": unique_nodes,
        "total_direct": len(requirements),
        "total_resolved": len(unique_nodes),
        "total_compressed_mb": round(total_compressed_mb, 2),
        "estimated_uncompressed_mb": round(estimated_uncompressed_mb, 2),
        "pct_of_fluid_limit": pct_of_fluid_limit,
        "verdict": verdict,
        "alternatives": alternatives,
        "fixed_requirements": fixed_requirements,
        "cold_start": cold_start,
        "route_split": route_split,
        "cuda_bloat": cuda_bloat,
        "corrected_estimate": corrected_estimate,
    }
