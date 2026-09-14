# Bundle Doctor 🩺

**Will your Python function actually fit on Vercel?**

[Live demo](https://vercel-python-bundle-doctor.vercel.app) · [Source](https://github.com/k-kj0/vercel-python-bundle-doctor)

## The problem

Vercel Python Functions cap out at **250MB** on the standard path, or **500MB** with Fluid Compute. Developers keep hitting this limit unexpectedly — because a `requirements.txt` only lists *direct* dependencies. Each of those pulls in transitive dependencies you never see until deploy time, and the gap between "looks small" and "actually deployed" is often 5-10x.

This is a real, currently unsolved pain point — Vercel's own community forums have developers confused by a requirements.txt reporting a few MB total that somehow produces a 60MB+ deployed function, with no clear explanation why. There is no official Vercel tool that resolves your full dependency tree and warns you *before* you deploy and fail.

## What it does

Paste a `requirements.txt`. Bundle Doctor:

- **Resolves the full transitive dependency tree** live from PyPI — not a static lookup table, every result is computed fresh on request by recursively walking each package's actual declared dependencies.
- **Selects the correct wheel variant** for Vercel's Linux/CPU runtime, avoiding GPU/CUDA builds that can inflate size estimates by 5-10x for packages like `torch`.
- **Detects CUDA/GPU bloat** specifically: if `torch` pulls in `nvidia-*` packages (which happens by default on PyPI, since the standard wheel assumes a GPU target), it flags exactly how much that's costing you and gives the precise fix — installing from PyTorch's dedicated CPU-only index.
- **Shows a corrected estimate** alongside the raw one, so a scary "6GB, will fail" number sits next to "actually ~180MB, will pass" once the CUDA fix is applied — you see both the mistake and the reality.
- **Estimates cold start impact** based on bundle size and the presence of compiled/native packages (numpy, torch, opencv, etc.), which add import-time overhead beyond raw size alone.
- **Suggests route splitting** when multiple large, independent packages are bundled into a single function — since Vercel bundles each Python function based on what that file imports, isolating heavy packages into separate route handlers keeps unrelated routes fast.
- **Recommends lighter alternatives** (pandas → polars, torch → onnxruntime, opencv-python → opencv-python-headless) with real, live-checked size savings from PyPI — not static advice.
- **Generates a fixed `requirements.txt`** with alternatives applied, ready to copy and use.

25 example scenarios across 6 categories (common frameworks, data/ML, vision/automation, databases/auth, documents/PDFs, and multi-package route-split demos) let you see the tool's range in one click, or paste your own requirements freely.

## Tech stack

Python · FastAPI · Jinja2 · httpx (async PyPI API calls) · deployed on Vercel Python Functions

## Architecture

No database, no external state — every analysis is computed live from PyPI on each request.

## Known limitations (stated honestly)

- **Uncompressed size is estimated via a 3x multiplier** on the compressed download size — a heuristic, since actual expansion ratios vary by package. It's a strong early-warning signal, not a byte-perfect guarantee.
- **Does not perform full pip-style dependency resolution.** Real `pip install` reconciles version conflicts across branches of the dependency tree; this tool walks each branch independently, which can occasionally over-count in edge cases with shared sub-dependencies pinned to different versions.
- **Does not account for native binary compilation differences** across Python versions or exact OS patch levels — it targets Vercel's general Linux/manylinux CPU runtime, not a specific minor version.
- **CUDA bloat detection is currently torch-specific.** Other GPU-dependent packages (some TensorFlow builds, JAX) aren't yet covered by the same explicit callout.

## Roadmap

- Resolve exact pip-equivalent dependency versions with proper conflict resolution
- Auto-generate a fixed `requirements.txt` that's directly pip-installable, not just illustrative
- Ship as a GitHub Action that comments the bundle-size delta directly on pull requests, catching this before a Vercel deploy is even attempted
- Extend CUDA/GPU bloat detection beyond torch to other GPU-dependent packages

## Why this exists

Built to solve a gap Vercel hasn't shipped a first-party tool for yet: resolving real bundle size, cold start risk, and route structure *before* you deploy, not after it fails.
