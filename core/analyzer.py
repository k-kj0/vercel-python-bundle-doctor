import httpx
import re
from typing import List, Dict, Set

VERCEL_STANDARD_LIMIT_MB = 250
VERCEL_FLUID_LIMIT_MB = 500
MAX_DEPTH = 3  # how deep to walk the dependency tree

ALTERNATIVES = {
    "pandas": "polars",
    "torch": "onnxruntime",
    "tensorflow": "tflite-runtime",
    "opencv-python": "opencv-python-headless",
    "scipy": None,
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

def extract_dep_name(requires_dist_entry: str):
    # Skip optional/extra-only dependencies (e.g. "aiohttp; extra == 'async'")
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
    chosen = wheels[0] if wheels else (files[0] if files else None)
    return chosen.get("size", 0) if chosen else 0

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

async def analyze_requirements(text: str) -> Dict:
    requirements = parse_requirements(text)
    visited: Set[str] = set()
    all_nodes: List[Dict] = []

    async with httpx.AsyncClient() as client:
        for req in requirements:
            name, version = split_name_version(req)
            all_nodes.extend(await resolve_tree(client, name, version, visited))

        # Dedupe (keep first occurrence, which is shallowest depth)
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

    verdict = "PASS"
    if estimated_uncompressed_mb > VERCEL_FLUID_LIMIT_MB:
        verdict = "FAIL — exceeds 500MB Fluid Compute limit"
    elif estimated_uncompressed_mb > VERCEL_STANDARD_LIMIT_MB:
        verdict = "WARN — exceeds 250MB standard limit, needs Fluid Compute opt-in"
    else:
        verdict = "PASS — within limits"

    pct_of_fluid_limit = min(100, round((estimated_uncompressed_mb / VERCEL_FLUID_LIMIT_MB) * 100, 1))

    return {
        "packages": unique_nodes,
        "total_direct": len(requirements),
        "total_resolved": len(unique_nodes),
        "total_compressed_mb": round(total_compressed_mb, 2),
        "estimated_uncompressed_mb": round(estimated_uncompressed_mb, 2),
        "pct_of_fluid_limit": pct_of_fluid_limit,
        "verdict": verdict,
        "alternatives": alternatives,
    }
