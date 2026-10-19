"""Fixed prefill-shape workloads and randomized paired triplet plans."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = ROOT/'new/scripts/prefill_confirmation/run_prefill_confirmation.py'
BASE_SHA256 = '981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75'
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
CONDITIONS = ('control', 'burst', 'distributed')
SETTINGS = dict(vllm='0.17.1', model=MODEL, cpu_offload_gb=0,
                prefix_caching=False, block_size=16, num_gpu_blocks=1024,
                max_model_len=4096, max_num_seqs=16, server_seed=20261017,
                metrics_interval_ms=50, max_metrics_gap_ms=250,
                max_dispatch_lateness_ms=50, request_timeout_s=60,
                episode_timeout_s=180)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_base():
    require(hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()==BASE_SHA256,
            'historical executor hash changed; review before reuse')
    spec=importlib.util.spec_from_file_location('prefill_shape_frozen_executor',BASE_PATH)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode=previous
    return module


def prepare(mode='diagnostic', *, study_id=None, seed=20261017,
            trigger_ms=None, interval_ms=None, tokens=None, diagnostic_budget=2048):
    require(mode in ('diagnostic','main'),'unknown study mode')
    require(type(seed) is int and seed>=0,'invalid master seed')
    tokens=list(range(100,132)) if tokens is None else tokens
    require(isinstance(tokens,list) and len(set(tokens))>=16 and
            all(type(t) is int and 0<=t<128000 for t in tokens),
            'public token vocabulary must contain >=16 distinct non-special base IDs')
    if mode=='diagnostic':
        require(type(diagnostic_budget) is int and diagnostic_budget in (512,2048),
                'diagnostic budget must be 512 or 2048')
        require(trigger_ms in (None,800) and interval_ms in (None,100),
                'diagnostic timer is fixed at 800 ms and spacing at 100 ms')
        trigger_ms,interval_ms,budgets,blocks=800,100,[diagnostic_budget],1
    else:
        require(diagnostic_budget==2048,'diagnostic budget is not a main-study selector')
        require(type(trigger_ms) is int and trigger_ms>0 and type(interval_ms) is int and interval_ms>0,
                'main study requires separately frozen trigger and spacing')
        budgets,blocks=[512,2048],10
    study_id=study_id or 'shape-'+uuid.uuid4().hex
    require(isinstance(study_id,str) and study_id and
            all(c.isalnum() or c in '-_' for c in study_id),'invalid study ID')
    episodes=[]
    for budget in budgets:
        for block in range(blocks):
            # Identical prompts and seeds across conditions AND paired budgets.
            prompt_seed=int(sha([seed,'prompts',block])[:16],16)
            rng=random.Random(prompt_seed)
            prompts={f'V{i}':[rng.choice(tokens) for _ in range(256)] for i in range(4)}
            prompts.update({f'E{i}':[rng.choice(tokens) for _ in range(2048)] for i in range(4)})
            seeds={role:int(sha([seed,block,role])[:8],16)%(2**31-1) for role in prompts}
            order=list(CONDITIONS)
            random.Random(sha([seed,'order',budget,block])).shuffle(order)
            for position,condition in enumerate(order):
                episode_id=f'{study_id}-b{budget}-k{block:02d}-{condition}'
                roles=[f'V{i}' for i in range(4)] + ([] if condition=='control' else [f'E{i}' for i in range(4)])
                episodes.append(dict(episode_id=episode_id,mode=mode,budget=budget,
                    block_id=f'b{budget}-k{block:02d}',paired_block=block,
                    order_in_block=position,condition=condition,trigger_ms=trigger_ms,
                    interval_ms=interval_ms,prompts={r:prompts[r] for r in roles},
                    seeds={r:seeds[r] for r in roles},
                    outputs={r:64 if r.startswith('V') else 16 for r in roles},
                    request_ids={r:episode_id+':'+r for r in roles}))
    return dict(schema='prefill-shape-plan-v1',mode=mode,study_id=study_id,master_seed=seed,
                token_ids=tokens,trigger_ms=trigger_ms,interval_ms=interval_ms,settings=SETTINGS,
                primary='max(receipt_time(V0..V3 token64)) - common_victim_start',
                episodes=episodes,blocks_per_budget=blocks,budgets=budgets,
                completion_request_count=20*blocks*len(budgets),
                historical_executor_sha256=BASE_SHA256)


def validate(plan):
    budget=2048
    if plan['mode']=='diagnostic':
        require(isinstance(plan['budgets'],list) and len(plan['budgets'])==1,
                'diagnostic plan requires exactly one budget')
        budget=plan['budgets'][0]
    expected=prepare(plan['mode'],study_id=plan['study_id'],seed=plan['master_seed'],
                     trigger_ms=plan['trigger_ms'],interval_ms=plan['interval_ms'],tokens=plan['token_ids'],
                     diagnostic_budget=budget)
    require(plan==expected,'plan differs from declared fixed design')
    return plan


def validate_episode(plan):
    require(plan['mode'] in ('diagnostic','main') and plan['condition'] in CONDITIONS and
            plan['budget'] in (512,2048),'invalid episode condition/mode/budget')
    require(plan['mode']!='diagnostic' or
            (plan['trigger_ms']==800 and plan['interval_ms']==100),'diagnostic design changed')
    roles={f'V{i}' for i in range(4)} | (set() if plan['condition']=='control' else {f'E{i}' for i in range(4)})
    require(set(plan['prompts'])==set(plan['outputs'])==set(plan['seeds'])==set(plan['request_ids'])==roles,
            'episode cohort differs from design')
    for role,prompt in plan['prompts'].items():
        require(len(prompt)==(256 if role.startswith('V') else 2048) and
                plan['outputs'][role]==(64 if role.startswith('V') else 16) and
                all(type(t) is int and 0<=t<128000 for t in prompt),'workload length/token mismatch')
    return plan
