#!/usr/bin/env python3
"""
victim_worker.py
Wird von run_experiment.py als Subprocess gestartet.
Läuft so lange, bis SIGTERM kommt, dann schreibt es das JSON und beendet sich.

duration_s  = gesamter Wallclock-Run (admission window + drain)
drain_s     = duration_s - window_secs  (Zeit zum sauberen Abschluss laufender Requests)
total_input_tokens  = alle gestarteten Requests (ok + fail), Input-Seite
total_output_tokens = nur erfolgreiche Requests, Output-Seite
"""
import asyncio, json, os, random, signal, string, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import httpx, numpy as np

BASE_URL   = os.environ["BASE_URL"]
ENDPOINT   = os.environ["ENDPOINT"]
API_KEY    = os.environ["API_KEY"]
MODEL      = os.environ["MODEL"]
ROLE       = os.environ["ROLE"]           # victim | attacker
CONDITION  = os.environ["CONDITION"]      # cond_a | cond_b
INPUT_LEN  = int(os.environ["INPUT_LEN"])
OUTPUT_LEN = int(os.environ["OUTPUT_LEN"])
CONC       = int(os.environ["CONCURRENCY"])
TEMP       = float(os.environ["TEMPERATURE"])
OUTFILE    = os.environ["OUTFILE"]        # absoluter Pfad zur JSON-Zieldatei
EXPERIMENT_ID   = os.environ["EXPERIMENT_ID"]
SERVER_LABEL    = os.environ["SERVER_LABEL"]
OFFLOAD_GB      = int(os.environ["OFFLOAD_GB"])
WINDOW_SECS     = int(os.environ["WINDOW_SECS"])
RUN_NO          = int(os.environ["RUN_NO"])
TIMEOUT_S  = 120
PREVIEW    = 120

stop_event = asyncio.Event()

def _rand_prompt(n):
    chars, words = n * 4, []
    while sum(len(w)+1 for w in words) < chars:
        words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3,10))))
    return " ".join(words)

def _stats(data):
    if not data:
        return {"mean":None,"median":None,"std":None,"p50":None,"p95":None,"p99":None}
    a = np.array(data, dtype=float)
    return {"mean":float(np.mean(a)),"median":float(np.median(a)),
            "std":float(np.std(a,ddof=1) if len(a)>1 else 0.0),
            "p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),
            "p99":float(np.percentile(a,99))}

def _error_text(e: BaseException) -> str:
    """Garantiert nie leeren Fehlertext — fällt auf Typname zurück."""
    msg = str(e).strip()
    return msg if msg else type(e).__name__

@dataclass
class Req:
    idx: int; input_len: int; target_output_len: int
    actual_output_len: int = 0; success: bool = False; error: str = ""
    start_time: float = 0.0; ttft_s: float = 0.0; ttft_ms: float = 0.0
    itl: List[float] = field(default_factory=list)
    decode_ms: float = 0.0; e2el_ms: float = 0.0; preview: str = ""
    @property
    def tpot_ms(self):
        return self.decode_ms/max(1,self.actual_output_len-1) if self.actual_output_len>1 and self.decode_ms>0 else None
    def to_dict(self):
        return {"request_idx":self.idx,"role":ROLE,"condition":CONDITION,
                "input_len":self.input_len,"target_output_len":self.target_output_len,
                "actual_output_len":self.actual_output_len,"request_success":self.success,
                "error_text":self.error,"start_time":self.start_time,
                "ttft_s":self.ttft_s,"ttft_ms":self.ttft_ms,"itl_sequence":self.itl,
                "decode_time_ms":self.decode_ms,"e2el_ms":self.e2el_ms,
                "tpot_ms":self.tpot_ms,"generated_text_preview":self.preview}

async def do_request(client, idx):
    r = Req(idx=idx, input_len=INPUT_LEN, target_output_len=OUTPUT_LEN,
            start_time=time.perf_counter())
    try:
        payload = {"model":MODEL,"messages":[{"role":"user","content":_rand_prompt(INPUT_LEN)}],
                   "max_tokens":OUTPUT_LEN,"temperature":TEMP,"stream":True}
        headers = {"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"}
        async with client.stream("POST", f"{BASE_URL}{ENDPOINT}",
                                  json=payload, headers=headers, timeout=TIMEOUT_S) as resp:
            resp.raise_for_status()
            first, prev_ts, tokens, text = True, None, 0, []
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                try: chunk = json.loads(d)
                except: continue
                now = time.perf_counter()
                tok = (chunk.get("choices",[{}])[0].get("delta",{}) or {}).get("content") or ""
                if tok:
                    tokens += 1; text.append(tok)
                    if first:
                        r.ttft_s = now-r.start_time; r.ttft_ms = r.ttft_s*1000
                        first = False; prev_ts = now
                    else:
                        if prev_ts: r.itl.append((now-prev_ts)*1000)
                        prev_ts = now
            end = time.perf_counter()
            r.actual_output_len = tokens
            r.e2el_ms = (end-r.start_time)*1000
            r.decode_ms = r.e2el_ms - r.ttft_ms
            r.preview = "".join(text)[:PREVIEW]
            r.success = True
    except asyncio.CancelledError:
        # Task wurde von außen abgebrochen (SIGTERM-Drain) — kein echter Fehler,
        # aber wir wollen einen sauberen Eintrag im JSON.
        r.error = "CancelledError: task aborted during drain"
        raise   # CancelledError muss weitergereicht werden
    except Exception as e:
        r.error = _error_text(e)
    return r

async def worker():
    results, sem, ctr, pending = [], asyncio.Semaphore(CONC), [0], []
    lock = asyncio.Lock()
    async def one():
        r = None
        try:
            async with sem:
                async with lock: ctr[0]+=1; idx=ctr[0]
                async with httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None)) as c:
                    r = await do_request(c, idx)
                results.append(r)
        except asyncio.CancelledError:
            # Task wurde während sem.acquire() oder do_request() gecancelt.
            # Wir erstellen einen minimalen Fail-Eintrag, damit failed-Count stimmt.
            if r is None:
                r = Req(idx=-1, input_len=INPUT_LEN, target_output_len=OUTPUT_LEN)
                r.error = "CancelledError: task cancelled before request started"
            results.append(r)
            # CancelledError nicht weiterreichen — gather soll sauber durchlaufen
        except Exception as e:
            # Unerwarteter Fehler außerhalb von do_request
            if r is None:
                r = Req(idx=-1, input_len=INPUT_LEN, target_output_len=OUTPUT_LEN)
            r.error = _error_text(e)
            results.append(r)

    while not stop_event.is_set():
        pending = [t for t in pending if not t.done()]
        for _ in range(CONC - len(pending)):
            if stop_event.is_set(): break
            pending.append(asyncio.create_task(one()))
        await asyncio.sleep(0.01)
    if pending:
        await asyncio.gather(*[t for t in pending if not t.done()], return_exceptions=True)
    return results

def write_json(results, t0, t_stop, t_end):
    """
    t0      : Perf-Counter beim Start des Workers
    t_stop  : Perf-Counter bei SIGTERM (Ende des Admission-Fensters)
    t_end   : Perf-Counter nach dem Drain (alle Requests abgeschlossen)
    """
    ok   = [r for r in results if r.success]
    fail = len(results) - len(ok)
    duration  = t_end  - t0
    drain     = t_end  - t_stop
    dur = max(duration, 1e-9)

    # input_tokens: alle gestarteten Requests (ok + fail mit idx > 0)
    # output_tokens: nur erfolgreiche
    in_tok  = sum(r.input_len for r in results if r.idx > 0)
    out_tok = sum(r.actual_output_len for r in ok)

    doc = {
        "experiment_id":EXPERIMENT_ID,"server_config_label":SERVER_LABEL,
        "model_name":MODEL,"offload_gb":OFFLOAD_GB,"condition":CONDITION,"role":ROLE,
        "concurrency":CONC,"input_len":INPUT_LEN,"output_len":OUTPUT_LEN,
        "temperature":TEMP,"window_secs":WINDOW_SECS,"run_no":RUN_NO,
        # Zeitfelder — klar getrennt
        "duration_s":round(duration, 3),        # Wallclock gesamt (admission + drain)
        "drain_s":round(drain, 3),              # nur Drain-Phase nach SIGTERM
        # Zählfelder
        "submitted":len(results),               # alle Req-Objekte (ok + fail + cancel)
        "completed":len(ok),
        "failed":fail,
        # Token-Zählung — asymmetrisch by design, daher explizit dokumentiert
        # total_input_tokens  = alle gestarteten Requests (Schätzung Input-Last)
        # total_output_tokens = nur erfolgreiche Requests (tatsächlich erzeugte Tokens)
        "total_input_tokens":in_tok,
        "total_output_tokens":out_tok,
        "request_throughput":round(len(ok)/dur, 4),
        "output_throughput":round(out_tok/dur, 4),
        "total_token_throughput":round((in_tok+out_tok)/dur, 4),
        "ttft_ms":_stats([r.ttft_ms for r in ok]),
        "tpot_ms":_stats([r.tpot_ms for r in ok if r.tpot_ms is not None]),
        "itl_ms": _stats([v for r in ok for v in r.itl]),
        "e2el_ms":_stats([r.e2el_ms for r in ok]),
        "individual_request_results":[r.to_dict() for r in results],
    }
    Path(OUTFILE).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTFILE).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"[{ROLE}] JSON geschrieben: {OUTFILE}  "
          f"({len(ok)} OK / {fail} FAIL / drain={drain:.1f}s)",
          flush=True)

async def main():
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    t_stop = t0  # wird überschrieben wenn SIGTERM kommt

    def on_sigterm():
        nonlocal t_stop
        t_stop = time.perf_counter()
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, on_sigterm)
    results = await worker()
    t_end = time.perf_counter()
    write_json(results, t0, t_stop, t_end)

if __name__ == "__main__":
    asyncio.run(main())
