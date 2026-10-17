#!/usr/bin/env python3
"""Bounded local /metrics recorder; sends no completion requests."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import socket
import time
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlsplit


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def validate_url(url):
    u = urlsplit(url)
    if u.scheme != 'http' or u.hostname != '127.0.0.1' or u.username or u.password or u.query or u.fragment or u.path != '/metrics':
        raise ValueError('use http://127.0.0.1:PORT/metrics')
    if u.port is None or not 1 <= u.port <= 65535:
        raise ValueError('explicit valid port required')
    return url


def clock_domain():
    boot = Path('/proc/sys/kernel/random/boot_id')
    return {'hostname': socket.gethostname(), 'boot_id': boot.read_text().strip() if boot.exists() else None,
            'clock': 'time.perf_counter_ns', 'cross_process_comparison_requires_same_host_boot': True}


class AsyncCapture:
    """Same raw sample contract for a concurrent episode; persist every scrape.

    The caller supplies the existing HTTP transport and append-only writer.
    No scrape is awaited on the token-trigger path.
    """
    def __init__(self, transport, url, writer, *, interval=.05, timeout=2.0):
        validate_url(url)
        if not math.isfinite(interval) or not .01 <= interval <= 1:
            raise ValueError('metrics interval must be between .01 and 1 seconds')
        self.transport, self.url, self.writer = transport, url, writer
        self.interval, self.timeout = interval, timeout
        self.samples = []
        self._stop = asyncio.Event()
        self._task = None

    async def sample(self):
        row = {'scrape_start_perf_ns': time.perf_counter_ns(),
               'http_status': None, 'raw_body': None, 'error': None}
        try:
            row['http_status'], row['raw_body'] = await self.transport.get_json(
                self.url, {}, self.timeout)
            if row['http_status'] != 200:
                row['error'] = 'http_error'
        except Exception as exc:
            row['error'] = type(exc).__name__
        row['response_received_perf_ns'] = time.perf_counter_ns()
        row['wall_time_ns'] = time.time_ns()
        self.samples.append(row)
        self.writer.emit('metrics_sample', sample=row)
        return row

    def start(self):
        async def loop():
            while not self._stop.is_set():
                await self.sample()
                try:
                    await asyncio.wait_for(self._stop.wait(), self.interval)
                except asyncio.TimeoutError:
                    pass
        self._task = asyncio.create_task(loop())

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            await self._task
        self.writer.emit('metrics_capture_end', sample_count=len(self.samples))


def capture(url, output, duration, interval, timeout=2.0, episode_id=None):
    validate_url(url)
    if not all(math.isfinite(x) for x in (duration, interval, timeout)) or not 0 < duration <= 3600 or not 0.1 <= interval <= 60 or not 0 < timeout <= 10:
        raise ValueError('invalid duration/interval/timeout')
    opener = build_opener(ProxyHandler({}), NoRedirect())
    headers = {}
    if os.environ.get('VLLM_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
    stop = [False]
    prior = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        prior[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    samples = errors = 0
    start = time.perf_counter_ns()
    domain = clock_domain()
    try:
        with Path(output).open('x', encoding='utf-8') as stream:
            def emit(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
            emit({'record_type': 'capture_start', 'clock_domain': domain, 'episode_id': episode_id,
                  'capture_start_perf_ns': start, 'wall_time_ns': time.time_ns(),
                  'duration_s': duration, 'interval_s': interval, 'completion_requests_sent': 0})
            try:
                deadline = time.monotonic() + duration
                while not stop[0] and time.monotonic() < deadline:
                    scrape = time.perf_counter_ns()
                    row = {'record_type': 'metrics_sample', 'scrape_start_perf_ns': scrape,
                           'episode_id': episode_id, 'http_status': None, 'raw_body': None, 'error': None}
                    try:
                        with opener.open(Request(url, headers=headers), timeout=min(timeout, max(0.001, deadline-time.monotonic()))) as response:
                            blob = response.read(2*1024*1024 + 1)
                            if len(blob) > 2*1024*1024:
                                raise ValueError('metrics_body_too_large')
                            row.update(http_status=response.status, raw_body=blob.decode('utf-8'))
                            if response.status != 200:
                                row['error'] = 'http_error'
                    except HTTPError as exc:
                        row.update(http_status=exc.code, error='http_error')
                        exc.close()
                    except (OSError, ValueError) as exc:
                        row['error'] = type(exc).__name__
                    row['response_received_perf_ns'] = time.perf_counter_ns()
                    row['wall_time_ns'] = time.time_ns()
                    samples += 1
                    errors += row['error'] is not None
                    emit(row)
                    remaining = deadline-time.monotonic()
                    if remaining > 0 and not stop[0]:
                        time.sleep(min(interval, remaining))
            finally:
                emit({'record_type': 'capture_end', 'capture_end_perf_ns': time.perf_counter_ns(),
                      'sample_count': samples, 'error_count': errors, 'interrupted': stop[0]})
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    return {'sample_count': samples, 'error_count': errors, 'output': str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000/metrics')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration-s', type=float, required=True)
    p.add_argument('--interval-s', type=float, default=0.25)
    p.add_argument('--episode-id')
    a = p.parse_args()
    try:
        result = capture(a.url, a.output, a.duration_s, a.interval_s, episode_id=a.episode_id)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps(result))
    return int(result['error_count'] > 0 or result['sample_count'] == 0)


if __name__ == '__main__':
    raise SystemExit(main())
