#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: Python package 'requests' fehlt. Installiere es mit: pip install requests")
    raise


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def median_or_none(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def p95_or_none(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = math.ceil(0.95 * len(ys)) - 1
    idx = max(0, min(idx, len(ys) - 1))
    return ys[idx]


def make_prompt(input_len: int, output_len: int, role: str) -> str:
    filler = " ".join(["calibration"] * input_len)
    return (
        f"{filler}\n\n"
        f"You are part of a controlled local systems measurement. "
        f"Role: {role}. "
        f"Generate a long neutral continuation of about {output_len} short words. "
        f"Do not answer with explanations. Keep producing simple neutral words until the generation limit is reached."
    )


def summarize_request_metrics(reqs: list[dict[str, Any]], role: str) -> dict[str, Any]:
    subset = [r for r in reqs if r.get("role") == role]

    def vals(key: str) -> list[float]:
        out = []
        for r in subset:
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(float(v))
        return out

    tpot = vals("tpot_ms")
    itl = vals("itl_median_ms")
    ttft = vals("ttft_ms")
    e2el = vals("e2el_ms")

    return {
        f"{role}_request_count": len(subset),
        f"{role}_success_count": sum(1 for r in subset if r.get("success") is True),
        f"{role}_failed_count": sum(1 for r in subset if r.get("success") is not True),
        f"{role}_median_tpot_ms": median_or_none(tpot),
        f"{role}_p95_tpot_ms": p95_or_none(tpot),
        f"{role}_median_itl_ms": median_or_none(itl),
        f"{role}_p95_itl_ms": p95_or_none(itl),
        f"{role}_median_ttft_ms": median_or_none(ttft),
        f"{role}_p95_ttft_ms": p95_or_none(ttft),
        f"{role}_median_e2el_ms": median_or_none(e2el),
        f"{role}_p95_e2el_ms": p95_or_none(e2el),
    }


def stream_chat_request(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    role: str,
    request_index: int,
    input_len: int,
    output_len: int,
    temperature: float,
    timeout_s: float,
    ignore_eos: bool,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/v1/chat/completions"

    prompt = make_prompt(input_len=input_len, output_len=output_len, role=role)

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": output_len,
        "stream": True,
    }

    # vLLM accepts this sampling parameter. If your server rejects it, rerun without --ignore-eos.
    if ignore_eos:
        payload["ignore_eos"] = True

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    start_perf = time.perf_counter()
    start_epoch = time.time()
    first_token_perf: float | None = None
    end_perf: float | None = None
    token_times: list[float] = []
    chunks: list[str] = []
    error: str | None = None
    http_status: int | None = None

    try:
        with requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(10, timeout_s),
        ) as resp:
            http_status = resp.status_code

            if resp.status_code != 200:
                body = resp.text[:2000]
                raise RuntimeError(f"HTTP {resp.status_code}: {body}")

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = obj.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")

                if content:
                    t = time.perf_counter()
                    if first_token_perf is None:
                        first_token_perf = t
                    token_times.append(t)
                    chunks.append(content)

        end_perf = time.perf_counter()

    except Exception as e:
        end_perf = time.perf_counter()
        error = repr(e)

    success = error is None and first_token_perf is not None and len(chunks) > 0

    ttft_ms: float | None = None
    e2el_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: list[float] = []

    if end_perf is not None:
        e2el_ms = (end_perf - start_perf) * 1000.0

    if first_token_perf is not None:
        ttft_ms = (first_token_perf - start_perf) * 1000.0

    if len(token_times) >= 2:
        itl_ms = [
            (token_times[i] - token_times[i - 1]) * 1000.0
            for i in range(1, len(token_times))
        ]
        tpot_ms = (token_times[-1] - token_times[0]) * 1000.0 / (len(token_times) - 1)

    return {
        "role": role,
        "request_index": request_index,
        "success": success,
        "error": error,
        "http_status": http_status,
        "start_epoch": start_epoch,
        "end_epoch": time.time(),
        "input_len_requested": input_len,
        "output_len_requested": output_len,
        "output_chunks_observed": len(chunks),
        "generated_text_chars": sum(len(c) for c in chunks),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itl_median_ms": median_or_none(itl_ms),
        "itl_p95_ms": p95_or_none(itl_ms),
        "e2el_ms": e2el_ms,
    }


def run_episode(args: argparse.Namespace, conc: int, condition: str, run_no: int, out_path: Path) -> dict[str, Any]:
    episode_id = (
        f"phaseA_{args.model_key}_offload{args.offload_gb}_"
        f"conc{conc}_{condition}_run{run_no}"
    )

    print(f"\n=== EPISODE {episode_id} ===", flush=True)

    max_workers = conc
    if condition == "fixed_burst":
        max_workers += args.burst_parallel

    request_results: list[dict[str, Any]] = []
    episode_start_perf = time.perf_counter()
    episode_start_epoch = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []

        for i in range(conc):
            futures.append(
                ex.submit(
                    stream_chat_request,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model_name=args.model_name,
                    role="victim",
                    request_index=i,
                    input_len=args.victim_input_len,
                    output_len=args.victim_output_len,
                    temperature=args.temperature,
                    timeout_s=args.timeout_s,
                    ignore_eos=args.ignore_eos,
                )
            )

        if condition == "fixed_burst":
            if args.burst_delay_s > 0:
                time.sleep(args.burst_delay_s)

            for i in range(args.burst_parallel):
                futures.append(
                    ex.submit(
                        stream_chat_request,
                        api_base=args.api_base,
                        api_key=args.api_key,
                        model_name=args.model_name,
                        role="load",
                        request_index=i,
                        input_len=args.burst_input_len,
                        output_len=args.burst_output_len,
                        temperature=args.temperature,
                        timeout_s=args.timeout_s,
                        ignore_eos=args.ignore_eos,
                    )
                )

        for fut in as_completed(futures):
            result = fut.result()
            request_results.append(result)
            print(
                f"  {result['role']:>6} #{result['request_index']:<2} "
                f"success={result['success']} "
                f"ttft={result.get('ttft_ms')} "
                f"tpot={result.get('tpot_ms')} "
                f"chunks={result.get('output_chunks_observed')}",
                flush=True,
            )

    episode_end_perf = time.perf_counter()
    episode_end_epoch = time.time()

    summary: dict[str, Any] = {
        "episode_id": episode_id,
        "experiment_id": "phase_a_fixed_burst",
        "timestamp_utc": now_iso(),
        "model_key": args.model_key,
        "model_name": args.model_name,
        "offload_gb": args.offload_gb,
        "concurrency": conc,
        "condition": condition,
        "run_no": run_no,
        "api_base": args.api_base,
        "temperature": args.temperature,
        "episode_start_epoch": episode_start_epoch,
        "episode_end_epoch": episode_end_epoch,
        "episode_duration_s": episode_end_perf - episode_start_perf,
        "victim_profile": {
            "input_len": args.victim_input_len,
            "output_len": args.victim_output_len,
        },
        "burst_profile": {
            "enabled": condition == "fixed_burst",
            "parallel_requests": args.burst_parallel if condition == "fixed_burst" else 0,
            "input_len": args.burst_input_len if condition == "fixed_burst" else None,
            "output_len": args.burst_output_len if condition == "fixed_burst" else None,
            "burst_delay_s": args.burst_delay_s if condition == "fixed_burst" else None,
        },
        "notes": "Controlled local Phase-A pilot: victim traffic vs fixed bounded load.",
    }

    summary.update(summarize_request_metrics(request_results, "victim"))
    summary.update(summarize_request_metrics(request_results, "load"))

    episode_json = {
        "metadata": summary,
        "requests": sorted(
            request_results,
            key=lambda r: (str(r.get("role")), int(r.get("request_index", 0))),
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(episode_json, f, indent=2)
    tmp_path.replace(out_path)

    print(
        f"Saved: {out_path}\n"
        f"  victim_median_tpot_ms={summary.get('victim_median_tpot_ms')}\n"
        f"  victim_median_itl_ms={summary.get('victim_median_itl_ms')}\n"
        f"  victim_median_ttft_ms={summary.get('victim_median_ttft_ms')}\n"
        f"  victim_success={summary.get('victim_success_count')}/{summary.get('victim_request_count')}",
        flush=True,
    )

    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    keys = sorted({k for r in rows for k in r.keys() if not isinstance(r.get(k), (dict, list))})
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            flat = {k: r.get(k) for k in keys}
            w.writerow(flat)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase-A no_burst vs fixed_burst episodes against a running vLLM server.")

    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="pilotkey")

    ap.add_argument("--model-key", default="llama")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--offload-gb", type=int, required=True)

    ap.add_argument("--concs", default="4,8")
    ap.add_argument("--conditions", default="no_burst,fixed_burst")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260711)

    ap.add_argument("--victim-input-len", type=int, default=256)
    ap.add_argument("--victim-output-len", type=int, default=64)

    ap.add_argument("--burst-parallel", type=int, default=4)
    ap.add_argument("--burst-input-len", type=int, default=256)
    ap.add_argument("--burst-output-len", type=int, default=256)
    ap.add_argument("--burst-delay-s", type=float, default=0.0)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--ignore-eos", action="store_true")

    ap.add_argument(
        "--out-root",
        default=None,
        help="Default: <project>/new/runs/phase_a_fixed_burst/<model_key>/offloadX",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing episode JSON files.")

    args = ap.parse_args()

    concs = parse_int_list(args.concs)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    allowed_conditions = {"no_burst", "fixed_burst"}

    for c in conditions:
        if c not in allowed_conditions:
            print(f"ERROR: unknown condition {c}. Allowed: {sorted(allowed_conditions)}")
            return 2

    if args.out_root is None:
        script_path = Path(__file__).resolve()
        # expected: DOS/new/scripts/script.py -> project root is parents[2]
        project_root = script_path.parents[2]
        out_root = project_root / "new" / "runs" / "phase_a_fixed_burst" / args.model_key / f"offload{args.offload_gb}"
    else:
        out_root = Path(args.out_root)

    episodes = []
    for conc in concs:
        for condition in conditions:
            for run_no in range(1, args.repeats + 1):
                episodes.append((conc, condition, run_no))

    rng = random.Random(args.seed + args.offload_gb)
    rng.shuffle(episodes)

    print("=== PHASE-A FIXED-BURST RUNNER ===")
    print(f"api_base: {args.api_base}")
    print(f"model_key: {args.model_key}")
    print(f"model_name: {args.model_name}")
    print(f"offload_gb metadata: {args.offload_gb}")
    print(f"concs: {concs}")
    print(f"conditions: {conditions}")
    print(f"repeats: {args.repeats}")
    print(f"victim: input={args.victim_input_len}, output={args.victim_output_len}")
    print(f"burst: parallel={args.burst_parallel}, input={args.burst_input_len}, output={args.burst_output_len}")
    print(f"ignore_eos: {args.ignore_eos}")
    print(f"out_root: {out_root}")
    print()

    summaries: list[dict[str, Any]] = []

    for conc, condition, run_no in episodes:
        filename = (
            f"{args.model_key}_offload{args.offload_gb}_"
            f"conc{conc}_{condition}_run{run_no}.json"
        )
        out_path = out_root / filename

        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"SKIP existing: {out_path}")
            continue

        summary = run_episode(args, conc=conc, condition=condition, run_no=run_no, out_path=out_path)
        summaries.append(summary)

    summary_csv = out_root / f"{args.model_key}_offload{args.offload_gb}_phase_a_summary.csv"
    write_summary_csv(summary_csv, summaries)

    print("\n=== DONE ===")
    print(f"new episodes written: {len(summaries)}")
    print(f"summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())