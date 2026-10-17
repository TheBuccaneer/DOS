#!/usr/bin/env python3
"""One bounded D/B overlap episode, or offline audit. Never starts a server."""
import argparse
import asyncio
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from cache_smoke import BASE_SHA256, load_base
from cache_eviction_control import (JsonlRecord, api_process_start_seconds, snapshot,
                                    verify_config, verify_server_identity,
                                    verify_server_log, verify_startup_time)
from metrics_contract import finalize_existing_sampler
from overlap_contract import (CONDITIONS, assess_episode, assessment_exit_code,
                               comparison_gates, make_plan, nuisance_from_log)
from record_metrics import AsyncCapture, clock_domain, validate_url


class ReceiptClock:
    """Preserve the actual wire receipt before synchronous durable journaling.

    The frozen executor reads this once immediately after each transport yield.
    All other readings use the same system perf_counter_ns clock.
    """
    def __init__(self, base):
        self.base, self.receipt = base, None

    def perf_counter_ns(self):
        if self.receipt is not None:
            value, self.receipt = self.receipt, None
            return value
        return time.perf_counter_ns()

    def utcnow_iso(self):
        return self.base.utcnow_iso()


class EpisodeTransport:
    def __init__(self, delegate, plan, role, writer, clock):
        self.delegate, self.plan, self.role = delegate, plan, role
        self.writer, self.clock, self.payload = writer, clock, None

    async def stream_completion(self, url, headers, payload, timeout_s, *, request_id=None):
        role = self.role
        if (payload['prompt'] != self.plan['prompts'][role]
                or payload['max_tokens'] != self.plan['outputs'][role]
                or payload['min_tokens'] != self.plan['outputs'][role]
                or payload['seed'] != self.plan['seed']):
            raise ValueError('overlap workload bounds changed')
        self.payload = dict(payload, cache_salt=self.plan['salts'][role])
        self.writer.emit('request_wire', request_id=request_id, role=role, payload=self.payload)
        stream = self.delegate.stream_completion(url, headers, self.payload, timeout_s, request_id=request_id)
        try:
            async for index, raw in enumerate_async(stream):
                received = time.perf_counter_ns()
                self.writer.emit('sse_wire', request_id=request_id, role=role,
                                 event_index=index, receive_perf_counter_ns=received, raw_data=raw)
                self.clock.receipt = received
                yield raw
        finally:
            await stream.aclose()


async def enumerate_async(stream):
    index = 0
    async for raw in stream:
        yield index, raw
        index += 1


def startup_evidence(path, config):
    sha, pid, log_start = verify_server_log(path, config['blocks'], 16, 256, config['budget'])
    process_start = api_process_start_seconds(pid)
    verify_startup_time(log_start, process_start)
    raw = path.read_text()
    if "'enable_prompt_tokens_details': True" not in raw:
        raise ValueError('restart with the updated starter: effective prompt token details required')
    # The original resolved config stays in raw; its fingerprint masks only
    # checked factor values and the directly budget-derived compile endpoint.
    nuisance = nuisance_from_log(raw, config)
    config['nuisance_sha256'] = hashlib.sha256(json.dumps(nuisance, sort_keys=True).encode()).hexdigest()
    domain = clock_domain()
    return {'startup_log': raw, 'startup_log_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'startup_log_path': str(path.resolve()),
            'server_identity': {'api_pid': pid, 'process_start_seconds': process_start,
                                'log_start_seconds': log_start, 'boot_id': domain['boot_id']},
            'clock_domain': domain, 'nuisance_config': nuisance,
            'verified_startup_log_sha256': sha}


async def run_episode(base, transport, plan, writer, evidence, *, base_url, api_key,
                      server_log, interval=.05, settle_s=5, request_timeout_s=60):
    ep = dict(evidence, plan=plan, requests={}, token_progress=[], triggers=[], errors=[],
              cohort_declared_perf_ns=time.perf_counter_ns(), metrics_interval_s=interval,
              historical_executor_sha256=BASE_SHA256)
    writer.emit('episode_start', episode=ep)
    sampler = AsyncCapture(transport, base_url + '/metrics', writer, interval=interval)
    tasks = []
    b_task = None
    log_stat = server_log.stat()
    log_offset = log_stat.st_size
    ep['iteration_log_byte_start'] = log_offset

    async def execute(role, created):
        clock = ReceiptClock(base.RealClock())
        tap = EpisodeTransport(transport, plan, role, writer, clock)
        record = await base.execute_completion_request(
            transport=tap, clock=clock, url=base_url + '/v1/completions', api_key=api_key,
            model_full_id=plan['config']['model'], prompt_token_ids=plan['prompts'][role],
            max_tokens=plan['outputs'][role], min_tokens=plan['outputs'][role], temperature=0,
            request_seed=plan['seed'], request_id=plan['request_ids'][role],
            role='victim' if role == 'D' else 'burst', request_index=0,
            prompt_seed=plan['seed'], generation_seed=plan['seed'],
            expected_prompt_tokens=len(plan['prompts'][role]),
            expected_completion_tokens=plan['outputs'][role], http_timeout_s=request_timeout_s,
            task_created_ns=created,
            on_output_tokens=progress if role == 'D' else None)
        record.update(cache_salt=plan['salts'][role], wire_payload=tap.payload)
        ep['requests'][role] = record
        writer.emit('request_result', role=role, record=record)
        return record

    def progress(count, received):
        nonlocal b_task
        previous = ep['token_progress'][-1]['count'] if ep['token_progress'] else 0
        ep['token_progress'].append({'count': count, 'receive_perf_counter_ns': received})
        if previous < 8 <= count:
            trigger = {'count': count, 'trigger_perf_ns': received,
                       'observed_perf_ns': time.perf_counter_ns()}
            ep['triggers'].append(trigger)
            if count == 8 and plan['condition'] != 'victim-only':
                created = time.perf_counter_ns()
                b_task = asyncio.create_task(execute('B', created))
                tasks.append(b_task)
            writer.emit('trigger', trigger=trigger)

    try:
        initial = snapshot(await sampler.sample())
        verify_config(initial, plan['config']['blocks'], 16)
        verify_server_identity(initial, evidence['server_identity']['process_start_seconds'])
        sampler.start()
        d_task = asyncio.create_task(execute('D', time.perf_counter_ns()))
        tasks.append(d_task)
        await d_task
        if b_task is not None:
            await b_task
        # Wait on the EXISTING background sampler, never scrape concurrently.
        deadline = time.monotonic() + settle_s
        terminal = max((r['stream_end_ns'] for r in ep['requests'].values()), default=0)
        expected = 1 if plan['condition'] == 'victim-only' else 2
        while time.monotonic() < deadline:
            latest = sampler.samples[-1]
            try:
                state = snapshot(latest)
                count = state['counters']['prefill_requests'] - initial['counters']['prefill_requests']
                if (latest['scrape_start_perf_ns'] >= terminal and count >= expected
                        and state['gauges']['running'] == state['gauges']['waiting'] == 0):
                    # Let the final scheduler log write and a further metric
                    # scrape arrive; both are retained rather than truncated.
                    await asyncio.sleep(max(interval * 2, .1))
                    break
            except ValueError:
                pass
            await asyncio.sleep(interval)
    except (Exception, asyncio.CancelledError) as exc:
        ep['errors'].append({'type': type(exc).__name__, 'message': str(exc).replace(api_key, '<redacted>')})
        writer.emit('episode_error', error=ep['errors'][-1])
    finally:
        await base.cancel_all(tasks)
        for role in ('D', 'B') if plan['condition'] != 'victim-only' else ('D',):
            ep['requests'].setdefault(role, {'request_id': plan['request_ids'][role],
                                             'request_index': 0, 'status': 'not_dispatched'})
        try:
            await finalize_existing_sampler(ep, sampler, clock_domain=ep['clock_domain'])
        except Exception as exc:
            ep['errors'].append({'type': type(exc).__name__, 'message': 'metrics finalization failed'})
        try:
            final_stat = server_log.stat()
            if ((final_stat.st_dev, final_stat.st_ino) != (log_stat.st_dev, log_stat.st_ino)
                    or final_stat.st_size < log_offset):
                ep['errors'].append({'type': 'LogChanged', 'message': 'server log replaced or truncated'})
            with server_log.open('rb') as stream:
                stream.seek(log_offset)
                segment = stream.read()
            ep.update(iteration_log=segment.decode('utf-8', errors='replace'),
                      iteration_log_byte_end=log_offset + len(segment),
                      iteration_log_sha256=hashlib.sha256(segment).hexdigest())
            writer.emit('server_log_segment', byte_start=log_offset,
                        byte_end=ep['iteration_log_byte_end'], raw_text=ep['iteration_log'],
                        sha256=ep['iteration_log_sha256'])
        except OSError as exc:
            ep['errors'].append({'type': type(exc).__name__, 'message': 'iteration log unavailable'})
        assessment = assess_episode(ep)
        writer.emit('episode_result', episode=ep, assessment=assessment)
    return ep, assessment


def read_episode(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    finals = [r for r in rows if r['record_type'] == 'episode_result']
    if len(finals) != 1 or rows[-1] is not finals[0]:
        raise ValueError('missing/duplicate final episode record; retain partial journal as invalid')
    ep = finals[0]['episode']
    starts = [r['episode'] for r in rows if r['record_type'] == 'episode_start']
    if len(starts) != 1 or any(starts[0][k] != ep[k] for k in (
            'plan', 'startup_log', 'startup_log_sha256', 'server_identity',
            'clock_domain', 'cohort_declared_perf_ns', 'nuisance_config')):
        raise ValueError('episode identity/startup evidence changed after declaration')
    if [r['trigger'] for r in rows if r['record_type'] == 'trigger'] != ep['triggers']:
        raise ValueError('trigger differs from durable journal')
    raw_metrics = [r['sample'] for r in rows if r['record_type'] == 'metrics_sample']
    if raw_metrics != ep['metrics_full_samples']:
        raise ValueError('final metrics differ from durable raw journal')
    for role, record in ep['requests'].items():
        if record['status'] == 'not_dispatched':
            continue
        records = [r['record'] for r in rows if r['record_type'] == 'request_result' and r['role'] == role]
        payloads = [r['payload'] for r in rows if r['record_type'] == 'request_wire' and r['role'] == role]
        if records != [record] or payloads != ([record['wire_payload']] if record['wire_payload'] is not None else []):
            raise ValueError('request result or salt differs from durable wire journal')
        wire = [r for r in rows if r['record_type'] == 'sse_wire' and r['role'] == role]
        if any(r['request_id'] != ep['plan']['request_ids'][role] or r['event_index'] != i
               for i, r in enumerate(wire)):
            raise ValueError('wire SSE request identity/index mismatch')
        if [r['raw_data'] for r in wire] != [e['raw_data'] for e in record['raw_sse_events']]:
            raise ValueError('SSE result differs from durable wire journal')
        if [r['receive_perf_counter_ns'] for r in wire] != [e['receive_perf_counter_ns'] for e in record['raw_sse_events']]:
            raise ValueError('SSE receipt timestamps differ from durable wire journal')
    segments = [r['raw_text'] for r in rows if r['record_type'] == 'server_log_segment']
    if segments != [ep.get('iteration_log')]:
        raise ValueError('scheduler log differs from durable journal')
    if any(r.get('role') not in ep['requests'] for r in rows if r['record_type'] in (
            'request_wire', 'request_result', 'sse_wire')):
        raise ValueError('unexpected additional request in journal')
    return ep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='sends at most two requests to an already running local server')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--prompt-token-ids', type=Path, required=True, help='public 128-token JSON array')
    run.add_argument('--condition', choices=CONDITIONS, required=True)
    run.add_argument('--server-log', type=Path, required=True)
    run.add_argument('--blocks', type=int, choices=(24, 64), required=True)
    run.add_argument('--budget', type=int, choices=(64, 512), required=True)
    run.add_argument('--base-url', default='http://127.0.0.1:8000')
    run.add_argument('--metrics-interval-s', type=float, default=.05)
    run.add_argument('--output', type=Path, required=True)
    audit = sub.add_parser('audit', help='offline raw-data and comparison gates only')
    audit.add_argument('episodes', type=Path, nargs='+')
    args = parser.parse_args()
    if args.command == 'audit':
        try:
            episodes = [read_episode(path) for path in args.episodes]
            result = comparison_gates(episodes)
            print(json.dumps(result, indent=2, allow_nan=False))
            return assessment_exit_code(result['episodes'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    base_url = args.base_url.rstrip('/')
    try:
        validate_url(base_url + '/metrics')
        if importlib.metadata.version('vllm') != '0.17.1':
            raise ValueError('vLLM 0.17.1 environment required')
        api_key = os.environ.get('VLLM_API_KEY')
        if not api_key:
            raise ValueError('VLLM_API_KEY required; never stored')
        base = load_base(args.root.resolve())
        # Exclusive output reservation precedes any network-capable operation.
        writer = JsonlRecord(args.output)
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))
    result = {'valid_episode': False}
    try:
        config = {'blocks': args.blocks, 'budget': args.budget, 'max_model_len': 256,
                  'model': 'meta-llama/Llama-3.1-8B-Instruct', 'metrics_interval_s': args.metrics_interval_s}
        evidence = startup_evidence(args.server_log, config)
        token_bytes = args.prompt_token_ids.read_bytes()
        nonce = uuid.uuid4().hex
        plan = make_plan(json.loads(token_bytes), args.condition, 'overlap-' + nonce, nonce, config)
        evidence.update(input_token_file_sha256=hashlib.sha256(token_bytes).hexdigest(),
                        input_token_file=str(args.prompt_token_ids.resolve()),
                        implementation_sha256={name: hashlib.sha256(
                            (Path(__file__).parent / name).read_bytes()).hexdigest() for name in (
                                'overlap_pilot.py', 'overlap_contract.py', 'record_metrics.py', 'run_server.sh')})
        transport = base.HttpxTransport()
        transport._httpx = SimpleNamespace(AsyncClient=functools.partial(
            transport._httpx.AsyncClient, trust_env=False))
        _, result = asyncio.run(run_episode(
            base, transport, plan, writer, evidence, base_url=base_url, api_key=api_key,
            server_log=args.server_log, interval=args.metrics_interval_s))
    except Exception as exc:
        writer.emit('preflight_error', error_type=type(exc).__name__,
                    error=str(exc).replace(api_key, '<redacted>'))
    finally:
        writer.close()
    print(json.dumps({'output': str(args.output), 'valid_episode': result['valid_episode'],
                      'technical_sse_measurement_complete': result.get('technical_sse_measurement_complete', False),
                      'mechanism_status': result.get('mechanism_status', 'unavailable'),
                      'causal_status': result.get('causal_status', 'ineligible_control_or_data_gates'),
                      'h1_status': 'not_evaluated', 'h1_validated': False}))
    return assessment_exit_code([result]) if 'mechanism_candidate' in result else 1


if __name__ == '__main__':
    raise SystemExit(main())
