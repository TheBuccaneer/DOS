#!/usr/bin/env python3
"""Read-only preflight metadata for the measurement computer.

Run with the existing experiment environment's Python. No model loads,
no HTTP requests, no package installation, no environment/credential dump.
Output describes availability and source occurrences, not live effectiveness.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

PACKAGES = ("vllm", "torch", "transformers", "httpx", "numpy", "pandas", "scipy")
SOURCES = (
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/sched/async_scheduler.py",
    "vllm/config/scheduler.py",
    "vllm/entrypoints/openai/completion/protocol.py",
    "vllm/v1/metrics/loggers.py",
)
TERMS = ("max_num_batched_tokens", "max_num_scheduled_tokens",
         "long_prefill_token_threshold", "max_num_partial_prefills",
         "max_long_partial_prefills", "cache_salt")


def collect(root, model_dir=None):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    sources = {}
    try:
        dist = metadata.distribution("vllm")
        for relative in SOURCES:
            path = Path(dist.locate_file(relative))
            if not path.is_file():
                sources[relative] = {"available": False}
                continue
            blob = path.read_bytes()
            lines = blob.decode("utf-8", errors="replace").splitlines()
            sources[relative] = {
                "available": True, "path": str(path),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "term_line_numbers": {term: [i for i, line in enumerate(lines, 1)
                                            if re.search(r"\b" + re.escape(term) + r"\b", line)]
                                      for term in TERMS},
            }
    except metadata.PackageNotFoundError:
        pass
    binary = shutil.which("nvidia-smi")
    gpu = {"available": bool(binary), "result": None}
    if binary:
        try:
            proc = subprocess.run([binary, "--query-gpu=name,memory.total,driver_version",
                                   "--format=csv,noheader"], capture_output=True, text=True,
                                  timeout=15, check=False)
            gpu.update({"exit_code": proc.returncode, "result": proc.stdout.strip(),
                        "stderr": proc.stderr.strip()[:400]})
        except (subprocess.TimeoutExpired, OSError) as exc:
            gpu["error"] = type(exc).__name__
    model = None
    if model_dir is not None:
        model_dir = model_dir.resolve()
        config = model_dir / "config.json"
        model = {"path": str(model_dir), "directory_exists": model_dir.is_dir(),
                 "config_exists": config.is_file(),
                 "top_level_weight_file_count": len(list(model_dir.glob("*.safetensors"))) + len(list(model_dir.glob("pytorch_model*.bin")))}
        if config.is_file():
            model["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_metadata", "experiment_ready": False,
        "readiness_note": "Runtime cache and budget effectiveness require separate measured evidence.",
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "hostname": platform.node(), "project_root": str(root.resolve()),
        "project_exists": root.is_dir(), "package_versions": versions,
        "gpu_metadata": gpu, "installed_sources": sources, "local_model": model,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.model_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
