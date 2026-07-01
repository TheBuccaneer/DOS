from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

DOS_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_ROOT = DOS_ROOT.parent
PORTO_ROOT = PROJECTS_ROOT / "porto"

OUT = DOS_ROOT / "results" / "defense_raw" / "raw_requests_0_vs_12.csv"

ROOTS = {
    "llama": PORTO_ROOT / "runs" / "2026-04-26_lama" / "base_runs",
    "qwen": PORTO_ROOT / "runs" / "2026-06-01_Qwen",
}

PAT = re.compile(r"(?:qwen_)?offload(?P<offload>\d+)_conc(?P<conc>\d+)_run(?P<run>\d+)\.json$")

rows = []

for dataset, root in ROOTS.items():
    if not root.exists():
        raise FileNotFoundError(root)

    for path in sorted(root.rglob("*.json")):
        m = PAT.search(path.name)
        if not m:
            continue

        offload = int(m.group("offload"))
        conc = int(m.group("conc"))
        run_id = int(m.group("run"))

        if offload not in (0, 12):
            continue

        with path.open() as f:
            data = json.load(f)

        ttfts = data.get("ttfts")
        itls = data.get("itls")
        output_lens = data.get("output_lens", [None] * len(ttfts))

        if ttfts is None or itls is None:
            print("SKIP missing ttfts/itls:", path)
            continue

        if len(ttfts) != len(itls):
            raise ValueError(f"Length mismatch in {path}: ttfts={len(ttfts)} itls={len(itls)}")

        for request_idx, (ttft_s, itl_seq_s) in enumerate(zip(ttfts, itls)):
            itl_ms = np.asarray(itl_seq_s, dtype=float) * 1000.0
            ttft_ms = float(ttft_s) * 1000.0

            if len(itl_ms) == 0:
                continue

            decode_time_ms = float(np.sum(itl_ms))

            rows.append({
                "dataset": dataset,
                "offload_gb": offload,
                "label": int(offload == 12),
                "run_concurrency": conc,
                "run_id": run_id,
                "request_idx": request_idx,
                "source_file": str(path.relative_to(PROJECTS_ROOT)),
                "model_name": data.get("model_name"),
                "input_len": int(data.get("input_len", 0)),
                "target_output_len": int(data.get("output_len", 0)),
                "actual_output_len": output_lens[request_idx] if request_idx < len(output_lens) else None,
                "ttft_ms": ttft_ms,
                "itl_count": int(len(itl_ms)),
                "itl_mean_ms": float(np.mean(itl_ms)),
                "itl_median_ms": float(np.median(itl_ms)),
                "itl_std_ms": float(np.std(itl_ms)),
                "itl_min_ms": float(np.min(itl_ms)),
                "itl_max_ms": float(np.max(itl_ms)),
                "itl_p95_ms": float(np.percentile(itl_ms, 95)),
                "itl_p99_ms": float(np.percentile(itl_ms, 99)),
                "decode_time_ms": decode_time_ms,
                "e2el_ms": ttft_ms + decode_time_ms,
            })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("saved:", OUT)
print("rows:", len(df))
print()
print(df.groupby(["dataset", "offload_gb", "run_concurrency"]).size().to_string())
print()
print(df.groupby(["dataset", "offload_gb"])[["ttft_ms", "itl_median_ms", "decode_time_ms", "e2el_ms"]].median().round(3).to_string())
