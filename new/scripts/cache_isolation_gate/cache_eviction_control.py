#!/usr/bin/env python3
"""Six sequential, synthetic cache-eviction requests. No server is started.

The append-only JSONL record survives a failed request or a missing metric.
One capacity arm is recorded per invocation; this script never claims H1.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from cache_smoke import BASE_SHA256, SaltTransport, load_base
from parse_iteration_logs import parse_text
from record_metrics import clock_domain, validate_url
from summarize_metrics import parse_metrics


ORDER = ('A1', 'A2', 'B1', 'B2', 'B3', 'A3')
SALT_A = 'icissp-eviction-control-A'
SALT_B = 'icissp-eviction-control-B'
COUNTERS = {
    'queries': 'vllm:prefix_cache_queries_total',
    'hits': 'vllm:prefix_cache_hits_total',
    'prompt_tokens': 'vllm:prompt_tokens_total',
    'computed_context_tokens': 'vllm:request_prefill_kv_computed_tokens_sum',
    'prefill_requests': 'vllm:request_prefill_kv_computed_tokens_count',
    'preemptions': 'vllm:num_preemptions_total',
}
GAUGES = {
    'running': 'vllm:num_requests_running',
    'waiting': 'vllm:num_requests_waiting',
}


class ControlError(ValueError):
    pass


def prompt_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def build_prompts(a_tokens):
    if not isinstance(a_tokens, list) or len(a_tokens) != 128 or any(
        type(t) is not int or t < 0 for t in a_tokens
    ):
        raise ControlError('A prompt must contain exactly 128 nonnegative integer token IDs')
    unique = list(dict.fromkeys(a_tokens))
    if len(unique) < 2:
        raise ControlError('A prompt needs at least two distinct token IDs to derive B blocks')
    x, y = unique[:2]
    candidates = ([x] * 16, [y] * 16, [x, y] * 8, [y, x] * 8,
                  [x] * 15 + [y], [y] * 15 + [x])
    first_blocks = []
    for block in candidates:
        if block != a_tokens[:16] and block not in first_blocks:
            first_blocks.append(block)
        if len(first_blocks) == 3:
            break
    if len(first_blocks) != 3:
        raise ControlError('could not derive three distinct B first blocks')
    b_prompts = [block + a_tokens[16:] for block in first_blocks]
    return dict(zip(ORDER, [a_tokens, a_tokens, *b_prompts, a_tokens]))


def _one_series(series, name):
    matches = [value for key, value in series.items()
               if key == name or key.startswith(name + '{')]
    if len(matches) != 1:
        raise ControlError(f'missing or ambiguous metric: {name}')
    return matches[0]


def _label(key, name):
    match = re.search(r'(?:\{|,)' + re.escape(name) + r'="([^"\\]*)"', key)
    if not match:
        raise ControlError(f'missing cache config label: {name}')
    return match.group(1)


def snapshot(sample):
    if sample.get('http_status') != 200:
        raise ControlError('metrics HTTP status is not 200')
    series = parse_metrics(sample.get('raw_body'))
    info = [key for key in series if key.startswith('vllm:cache_config_info{')]
    if len(info) != 1 or series[info[0]] != 1:
        raise ControlError('missing or ambiguous cache_config_info')
    labels = {name: _label(info[0], name) for name in (
        'num_gpu_blocks', 'num_gpu_blocks_override', 'block_size',
        'enable_prefix_caching', 'cpu_offload_gb', 'kv_cache_memory_bytes')}
    return {
        'counters': {key: _one_series(series, metric) for key, metric in COUNTERS.items()},
        'gauges': {key: _one_series(series, metric) for key, metric in GAUGES.items()},
        'process_start_time_seconds': _one_series(series, 'process_start_time_seconds'),
        'cache_config': labels,
    }


def verify_config(state, expected_blocks, expected_block_size):
    config = state['cache_config']
    expected = {
        'num_gpu_blocks': str(expected_blocks),
        'num_gpu_blocks_override': str(expected_blocks),
        'block_size': str(expected_block_size),
        'enable_prefix_caching': 'True',
        'cpu_offload_gb': '0',
        'kv_cache_memory_bytes': 'None',
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ControlError(f'cache configuration mismatch: {key}={config[key]!r}, expected {value!r}')
    if state['gauges']['running'] != 0 or state['gauges']['waiting'] != 0:
        raise ControlError('server is not idle at metrics boundary')


def metric_delta(before, after):
    result = {}
    for key, old in before['counters'].items():
        new = after['counters'][key]
        if new < old:
            raise ControlError(f'metric counter reset: {key}')
        result[key] = new - old
    for key in ('queries', 'prompt_tokens'):
        if result[key] > 128:
            raise ControlError(f'unexpected additional request or counter change: {key}')
    if result['prefill_requests'] > 1 or result['preemptions'] > 0:
        raise ControlError('unexpected additional prefill or preemption')
    if result['hits'] > 128 or result['computed_context_tokens'] > 128:
        raise ControlError('cache metrics exceed one 128-token request')
    complete = (result['queries'] == 128 and result['prompt_tokens'] == 128
                and result['prefill_requests'] == 1)
    if complete:
        if result['hits'] + result['computed_context_tokens'] != 128:
            raise ControlError('hit and computed-context counters do not reconcile')
        if any(value != int(value) for value in result.values()):
            raise ControlError('fractional token/request counter delta')
    return result, complete


def api_process_start_seconds(pid):
    """Read the Linux API process birth time independently of its metrics."""
    stat = Path(f'/proc/{pid}/stat').read_text()
    ticks = int(stat.rsplit(')', 1)[1].split()[19])
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    return boot + ticks / os.sysconf('SC_CLK_TCK')


def verify_server_identity(state, expected_process_start):
    observed = state['process_start_time_seconds']
    if abs(observed - expected_process_start) > 2:
        raise ControlError('metrics endpoint belongs to a different API server process')


def verify_startup_time(log_start, process_start):
    if not log_start - 5 <= process_start <= log_start + 300:
        raise ControlError('API server process start does not match startup log time')


def verify_server_log(path, blocks, block_size, max_model_len, budget):
    body = path.read_text(encoding='utf-8')
    required = (
        f'Max model len requested: {max_model_len}',
        f'Max num batched tokens (chunked-prefill budget): {budget}',
        f'NUM_GPU_BLOCKS_OVERRIDE requested: {blocks}',
        'Prefix-Caching: enabled',
        'Offload-Wert: 0',
    )
    lines = body.splitlines()
    for item in required:
        if not any(line == item or line.startswith(item + ';') for line in lines):
            raise ControlError(f'server startup log lacks requested setting: {item}')
    if not re.search(r'^vLLM-Version: .*\b0\.17\.1\b', body, re.MULTILINE):
        raise ControlError('server startup log lacks vLLM 0.17.1 version')
    if not re.search(r'Chunked prefill is enabled with max_num_batched_tokens='
                     + str(budget) + r'\.', body):
        raise ControlError('server startup log lacks effective prefill budget confirmation')
    if not re.search(r'Overriding num_gpu_blocks=\d+ with num_gpu_blocks_override='
                     + str(blocks) + r'\b', body):
        raise ControlError('server startup log lacks effective block override confirmation')
    sizes = [int(value.replace(',', '')) for value in re.findall(
        r'GPU KV cache size: ([\d,]+) tokens', body)]
    if sizes != [blocks * block_size]:
        raise ControlError(f'startup KV token capacity {sizes} differs from blocks x block size')
    if not re.search(r'Maximum concurrency for ' + format(max_model_len, ',')
                     + r' tokens per request:', body):
        raise ControlError('server startup log lacks effective max model length confirmation')
    api_pids = set(re.findall(r'\(APIServer pid=(\d+)\)', body))
    if len(api_pids) != 1:
        raise ControlError('server startup log lacks a unique API server PID')
    log_times = re.findall(r'^UTC-Startzeit: (\d{8}T\d{6}Z)$', body, re.MULTILINE)
    if len(log_times) != 1:
        raise ControlError('server startup log lacks a unique UTC start time')
    log_start = datetime.strptime(log_times[0], '%Y%m%dT%H%M%SZ').replace(
        tzinfo=timezone.utc).timestamp()
    return hashlib.sha256(body.encode()).hexdigest(), int(next(iter(api_pids))), log_start


class JsonlRecord:
    def __init__(self, path):
        self.file = path.open('x', encoding='utf-8')

    def emit(self, kind, **fields):
        row = {'record_type': kind, 'perf_counter_ns': time.perf_counter_ns(),
               'wall_time_ns': time.time_ns(), **fields}
        self.file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


async def run_control(base, transport, prompts, writer, *, base_url, api_key, model,
                      expected_blocks, expected_block_size=16, sleep=asyncio.sleep,
                      max_metric_polls=20, server_log=None, expected_process_start=None):
    assessments = {}
    current = None
    try:
        for index, name in enumerate(ORDER):
            current = name
            tokens = prompts[name]
            salt = SALT_A if name.startswith('A') else SALT_B
            before_status, before_body = await transport.get_json(base_url + '/metrics', {}, 2.0)
            before = {'http_status': before_status, 'raw_body': before_body}
            writer.emit('metrics_before', request=name, sample=before)
            before_state = snapshot(before)
            verify_config(before_state, expected_blocks, expected_block_size)
            if expected_process_start is not None:
                verify_server_identity(before_state, expected_process_start)
            if index == 0 and any(before_state['counters'].values()):
                raise ControlError('A1 requires a fresh server with zero request/cache counters')
            writer.emit('request_start', request=name, index=index, salt=salt,
                        prompt_length=len(tokens), prompt_sha256=prompt_hash(tokens))
            log_start = server_log.stat().st_size if server_log is not None else None
            record = await base.execute_completion_request(
                transport=SaltTransport(transport, salt), clock=base.RealClock(),
                url=base_url + '/v1/completions', api_key=api_key,
                model_full_id=model, prompt_token_ids=tokens, max_tokens=2, min_tokens=2,
                temperature=0.0, request_seed=20261010, request_id='icissp-eviction-' + name,
                role='cache_eviction_control', request_index=index, prompt_seed=20261010,
                generation_seed=20261010, expected_prompt_tokens=128,
                expected_completion_tokens=2, http_timeout_s=60.0)
            writer.emit('request_result', request=name, salt=salt, record=record)
            if record.get('status') != 'complete' or len(record.get('output_token_ids', [])) != 2:
                raise ControlError(f'incomplete SSE request: {name}')
            delta = None
            after_state = None
            for poll in range(max_metric_polls):
                after_status, after_body = await transport.get_json(base_url + '/metrics', {}, 2.0)
                after = {'http_status': after_status, 'raw_body': after_body}
                writer.emit('metrics_after_poll', request=name, poll=poll, sample=after)
                after_state = snapshot(after)
                verify_config(after_state, expected_blocks, expected_block_size)
                if expected_process_start is not None:
                    verify_server_identity(after_state, expected_process_start)
                delta, complete = metric_delta(before_state, after_state)
                if complete:
                    break
                if poll + 1 < max_metric_polls:
                    await sleep(0.25)
            else:
                raise ControlError(f'cache metrics did not settle for {name}')
            expected_hits = {'A1': 0, 'A2': 112, 'B1': 0, 'B2': 0, 'B3': 0}.get(name)
            if expected_hits is not None and delta['hits'] != expected_hits:
                raise ControlError(f'{name} cache-hit prerequisite failed: observed {delta["hits"]}, expected {expected_hits}')
            iteration = {'context_tokens': None, 'reason': 'server_log_not_supplied'}
            if server_log is not None:
                with server_log.open('rb') as stream:
                    stream.seek(log_start)
                    segment = stream.read()
                parsed = parse_text(segment.decode('utf-8', errors='replace'))
                unambiguous = not parsed['errors'] and all(
                    row['context_requests'] <= 1 and row['generation_requests'] <= 1
                    for row in parsed['records'])
                iteration = {'log_byte_start': log_start,
                             'log_byte_end': log_start + len(segment),
                             'segment_sha256': hashlib.sha256(segment).hexdigest(),
                             'context_tokens': sum(row['context_tokens'] for row in parsed['records'])
                             if unambiguous else None,
                             'reason': None if unambiguous else 'missing_or_ambiguous_iteration_lines',
                             'parser_errors': parsed['errors']}
                if iteration['context_tokens'] is not None and iteration['context_tokens'] != delta['computed_context_tokens']:
                    raise ControlError(f'iteration log and computed-context metric disagree for {name}')
            assessments[name] = delta
            writer.emit('request_assessment', request=name, cache_hit_delta=delta['hits'],
                        computed_context_tokens=delta['computed_context_tokens'],
                        counter_deltas=delta, iteration_log=iteration,
                        metric_attribution='sequential_idle_boundaries')
        return True, assessments
    except Exception as exc:
        writer.emit('control_error', request=current, error_type=type(exc).__name__,
                    error=str(exc))
        return False, assessments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prompt-token-ids', type=Path, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--expected-num-gpu-blocks', type=int, required=True)
    parser.add_argument('--expected-block-size', type=int, default=16)
    parser.add_argument('--expected-max-model-len', type=int, required=True)
    parser.add_argument('--expected-budget', type=int, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--output', type=Path, required=True, help='new append-only JSONL result')
    args = parser.parse_args()
    try:
        if any(value < 1 for value in (args.expected_num_gpu_blocks,
                                      args.expected_block_size, args.expected_max_model_len,
                                      args.expected_budget)):
            raise ControlError('expected configuration values must be positive')
        if args.expected_num_gpu_blocks < 2:
            raise ControlError('block count must leave room for the null block')
        base_url = args.base_url.rstrip('/')
        validate_url(base_url + '/metrics')
        token_bytes = args.prompt_token_ids.read_bytes()
        prompts = build_prompts(json.loads(token_bytes))
        if not os.environ.get('VLLM_API_KEY'):
            raise ControlError('VLLM_API_KEY must be set; its value is not recorded')
        if importlib.metadata.version('vllm') != '0.17.1':
            raise ControlError('local vLLM package is not 0.17.1')
        base = load_base(args.root.resolve())
        writer = JsonlRecord(args.output)
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))
    valid = False
    assessments = {}
    try:
        log_sha, api_pid, log_start = verify_server_log(
            args.server_log, args.expected_num_gpu_blocks, args.expected_block_size,
            args.expected_max_model_len, args.expected_budget)
        process_start = api_process_start_seconds(api_pid)
        verify_startup_time(log_start, process_start)
        writer.emit('run_start', mode='sequential_kv_eviction_control', order=ORDER,
                    clock_domain=clock_domain(), historical_executor_sha256=BASE_SHA256,
                    local_vllm_version='0.17.1', model=args.model,
                    input_token_file=str(args.prompt_token_ids),
                    input_token_file_sha256=hashlib.sha256(token_bytes).hexdigest(),
                    prompts={name: {'token_ids': tokens, 'prompt_sha256': prompt_hash(tokens),
                                    'length': len(tokens)} for name, tokens in prompts.items()},
                    derivation='B first 16 IDs use deterministic patterns from first two distinct A IDs; remaining 112 IDs equal A',
                    server_log=str(args.server_log), server_log_sha256_at_start=log_sha,
                    api_server_pid=api_pid, api_process_start_seconds=process_start,
                    launcher_log_start_seconds=log_start,
                    requested_config={'num_gpu_blocks_override': args.expected_num_gpu_blocks,
                                      'block_size': args.expected_block_size,
                                      'max_model_len': args.expected_max_model_len,
                                      'max_num_batched_tokens': args.expected_budget,
                                      'cpu_offload_gb': 0})
        transport = base.HttpxTransport()
        transport._httpx = SimpleNamespace(
            AsyncClient=functools.partial(transport._httpx.AsyncClient, trust_env=False))
        valid, assessments = asyncio.run(run_control(
            base, transport, prompts, writer, base_url=base_url,
            api_key=os.environ['VLLM_API_KEY'], model=args.model,
            expected_blocks=args.expected_num_gpu_blocks,
            expected_block_size=args.expected_block_size, server_log=args.server_log,
            expected_process_start=process_start))
        writer.emit('run_end', status='valid_control_run' if valid else 'invalid_control_run',
                    request_count=len(assessments), assessments=assessments,
                    mechanism_validated=False, scheduler_interaction_validated=False,
                    server_log_sha256_at_end=hashlib.sha256(args.server_log.read_bytes()).hexdigest())
    except Exception as exc:
        writer.emit('control_error', request=None, error_type=type(exc).__name__, error=str(exc))
        writer.emit('run_end', status='invalid_control_run', request_count=len(assessments),
                    assessments=assessments, mechanism_validated=False,
                    scheduler_interaction_validated=False)
    finally:
        writer.close()
    print(json.dumps({'output': str(args.output), 'valid_control_run': valid}))
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
