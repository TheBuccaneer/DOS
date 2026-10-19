#!/usr/bin/env python3
"""Explicit manual server starter for one budget. No experiment is run here."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from design import MODEL,SETTINGS,require


def command(budget):
    require(budget in (512,2048),'budget must be 512 or 2048')
    return [sys.executable,'-m','vllm.entrypoints.cli.main','serve',MODEL,
        '--served-model-name',MODEL,'--host','127.0.0.1','--port','8018',
        '--dtype','bfloat16','--generation-config','vllm','--gpu-memory-utilization','0.90',
        '--tensor-parallel-size','1','--max-model-len','4096','--cpu-offload-gb','0',
        '--no-enable-prefix-caching','--enable-chunked-prefill',
        '--max-num-batched-tokens',str(budget),'--max-num-seqs','16',
        '--max-num-partial-prefills','1','--max-long-partial-prefills','1',
        '--long-prefill-token-threshold','0','--block-size','16',
        '--num-gpu-blocks-override','1024','--enable-logging-iteration-details','--seed','20261017']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget',type=int,choices=(512,2048),required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    config=dict(SETTINGS,budget=a.budget,host='127.0.0.1',port=8018)
    args=command(a.budget)
    if a.dry_run:
        print(json.dumps(dict(config=config,command=args,server_started=False),indent=2))
        return 0
    try:
        require(importlib.metadata.version('vllm')=='0.17.1','vLLM 0.17.1 required')
        require(bool(os.environ.get('VLLM_API_KEY')),'VLLM_API_KEY required; keep value out of command arguments')
        require(os.environ.get('CUDA_VISIBLE_DEVICES','0')=='0','this study uses CUDA device 0')
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',VLLM_HOST_IP='127.0.0.1',
                   PYTHONDONTWRITEBYTECODE='1',VLLM_SERVER_DEV_MODE='1')
        a.log.parent.mkdir(parents=True,exist_ok=True)
        with a.log.open('x') as log:
            def write(line):
                log.write(line);log.flush()
                sys.stdout.write(line);sys.stdout.flush()
            write('SHAPE_CONFIG_JSON: '+json.dumps(config,sort_keys=True)+'\n')
            write('SHAPE_VLLM_VERSION: 0.17.1\n')
            write('SHAPE_START_UTC_NS: '+str(time.time_ns())+'\n')
            write('SHAPE_COMMAND_JSON: '+json.dumps(args)+'\n')
            child=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,bufsize=1,start_new_session=True)
            write('SHAPE_API_PID: '+str(child.pid)+'\n')
            try:
                for line in child.stdout: write(line)
                return child.wait()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
    except (ValueError,OSError,importlib.metadata.PackageNotFoundError) as exc:
        p.exit(1,type(exc).__name__+': '+str(exc)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
