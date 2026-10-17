#!/usr/bin/env python3
"""Three tiny SEQUENTIAL local requests via the existing DOS Raw-SSE executor.

No parallel workload, pressure calibration or interference campaign.
"""
import argparse
import asyncio
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from record_metrics import validate_url, clock_domain

BASE_SHA256 = '981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75'


def load_base(root):
    path = root/'new/scripts/prefill_confirmation/run_prefill_confirmation.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('historical executor changed; review source before reuse')
    spec = importlib.util.spec_from_file_location('dos_gate_historical_executor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SaltTransport:
    def __init__(self, delegate, salt):
        self.delegate, self.salt = delegate, salt

    async def stream_completion(self, url, headers, payload, timeout_s, *, request_id=None):
        if not 1 <= len(payload.get('prompt', [])) <= 128 or payload.get('max_tokens') != 2 or payload.get('min_tokens') != 2:
            raise ValueError('function-check bounds violated')
        changed = dict(payload, cache_salt=self.salt)
        async for event in self.delegate.stream_completion(url, headers, changed, timeout_s, request_id=request_id):
            yield event


async def smoke(base, tokens, base_url, api_key, model):
    transport = base.HttpxTransport()
    # Reuse its methods and explicitly disable environmental HTTP proxies.
    transport._httpx = SimpleNamespace(AsyncClient=functools.partial(transport._httpx.AsyncClient, trust_env=False))
    rows = []
    for index, salt in enumerate(('icissp-function-tenant-A', 'icissp-function-tenant-A', 'icissp-function-tenant-B')):
        before_status, before = await transport.get_json(base_url+'/metrics', {}, 2.0)
        clock = base.RealClock()
        record = await base.execute_completion_request(
            transport=SaltTransport(transport, salt), clock=clock, url=base_url+'/v1/completions',
            api_key=api_key, model_full_id=model, prompt_token_ids=tokens, max_tokens=2, min_tokens=2,
            temperature=0.0, request_seed=20261010, request_id='icissp-cache-smoke-'+str(index),
            role='cache_function_check', request_index=index, prompt_seed=20261010,
            generation_seed=20261010, expected_prompt_tokens=len(tokens), expected_completion_tokens=2,
            http_timeout_s=30.0)
        await asyncio.sleep(0.5)
        after_status, after = await transport.get_json(base_url+'/metrics', {}, 2.0)
        rows.append({'request': record, 'cache_salt': salt,
                     'metrics_before': {'http_status': before_status, 'raw_body': before},
                     'metrics_after': {'http_status': after_status, 'raw_body': after}})
        if record.get('status') != 'complete':
            break
    return {'mode': 'bounded_sequential_cache_function_check', 'clock_domain': clock_domain(),
            'historical_executor_sha256': BASE_SHA256, 'request_count': len(rows), 'requests': rows,
            'cache_rule_validated': False, 'runtime_gate_passed': False,
            'note': 'Inspect actual cache-counter updates; scrape lag and pre-existing cache state can make this inconclusive.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prompt-token-ids', type=Path, required=True, help='JSON integer array, public test prefix only, 1..128 tokens')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        base_url = a.base_url.rstrip('/')
        validate_url(base_url+'/metrics')
        tokens = json.loads(a.prompt_token_ids.read_text())
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 128 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('prompt must contain 1..128 nonnegative integer token ids')
        if not os.environ.get('VLLM_API_KEY'):
            raise ValueError('VLLM_API_KEY must be set; value is not recorded')
        base = load_base(a.root.resolve())
        # Reserve the output before sending anything; never overwrite an existing file.
        with a.output.open('x') as file:
            try:
                result = asyncio.run(smoke(base, tokens, base_url, os.environ['VLLM_API_KEY'], a.model))
            except Exception as exc:
                result = {'mode': 'bounded_sequential_cache_function_check', 'status': 'failed',
                          'error_type': type(exc).__name__, 'runtime_gate_passed': False}
            json.dump(result, file, indent=2, ensure_ascii=False)
            file.write('\n')
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(json.dumps({'output': str(a.output), 'runtime_gate_passed': False}))
    return int(result.get('status') == 'failed' or result.get('request_count') != 3 or
               any(r['request'].get('status') != 'complete' for r in result.get('requests', [])))


if __name__ == '__main__':
    raise SystemExit(main())
