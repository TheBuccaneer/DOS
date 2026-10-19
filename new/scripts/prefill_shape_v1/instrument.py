"""Durable raw journaling, continuous metrics and receipt/send timestamps."""
import asyncio
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from design import require


def local_url(base_url):
    u=urlsplit(base_url)
    require(u.scheme=='http' and u.hostname=='127.0.0.1' and u.port is not None and
            1<=u.port<=65535 and not u.username and not u.password and
            u.path in ('','/') and not u.query and not u.fragment,'explicit loopback HTTP URL required')
    return base_url.rstrip('/')


class Journal:
    """One ordered fsync worker; disk I/O never runs inside a timer callback."""
    def __init__(self,path):
        self.file=Path(path).open('x',encoding='utf-8')
        self.queue=queue.Queue(maxsize=4096)
        self.sequence=0
        self.error=None
        self.closed=False
        self.thread=threading.Thread(target=self._write,daemon=True)
        self.thread.start()

    def _write(self):
        while True:
            row=self.queue.get()
            try:
                if row is None: return
                if self.error is None:
                    self.file.write(json.dumps(row,allow_nan=False)+'\n')
                    self.file.flush()
                    os.fsync(self.file.fileno())
            except Exception as exc:
                self.error=type(exc).__name__
            finally:
                self.queue.task_done()

    def emit(self,kind,**fields):
        require(not self.closed and self.error is None,'journal unavailable')
        row=dict(record_type=kind,journal_seq=self.sequence,perf_counter_ns=time.perf_counter_ns(),
                 wall_time_ns=time.time_ns(),**fields)
        try: self.queue.put_nowait(row)
        except queue.Full:
            self.error='QueueFull'
            raise ValueError('journal queue full')
        self.sequence+=1

    async def flush(self):
        # Avoid creating a second executor just to wait for our dedicated writer.
        # Poll only after an episode; the timer/receipt path never waits here.
        while True:
            with self.queue.mutex: pending=self.queue.unfinished_tasks
            if not pending: break
            await asyncio.sleep(.001)
        require(self.error is None,'journal persistence failure')

    def close(self):
        if not self.closed:
            self.closed=True
            self.queue.put(None)
            self.thread.join()
            self.file.close()
        require(self.error is None,'journal persistence failure')


class Sampler:
    def __init__(self,transport,url,journal,api_key,interval=.05):
        self.transport,self.url,self.journal=transport,url,journal
        self.headers={'Authorization':'Bearer '+api_key}
        self.interval=interval
        self.samples=[]
        self.stop_event=asyncio.Event()
        self.task=None

    async def sample(self):
        row=dict(scrape_start_ns=time.perf_counter_ns(),http_status=None,raw_body=None,error=None)
        try:
            row['http_status'],row['raw_body']=await self.transport.get_json(self.url,self.headers,2)
            if row['http_status']!=200: row['error']='http_error'
        except Exception as exc: row['error']=type(exc).__name__
        row['response_ns']=time.perf_counter_ns()
        self.samples.append(row)
        self.journal.emit('metrics_sample',sample=row)
        return row

    def start(self):
        async def loop():
            while not self.stop_event.is_set():
                await self.sample()
                try: await asyncio.wait_for(self.stop_event.wait(),self.interval)
                except asyncio.TimeoutError: pass
        self.task=asyncio.create_task(loop())

    async def stop(self):
        self.stop_event.set()
        if self.task is not None: await self.task


async def sleep_until(deadline,*,now=time.perf_counter_ns,sleep=asyncio.sleep):
    while True:
        remaining=(deadline-now())/1e9
        if remaining<=0: return
        await sleep(remaining)


class ReceiptClock:
    def __init__(self,base):
        self.base,self.receipt=base,None
    def utcnow_iso(self): return self.base.utcnow_iso()
    def perf_counter_ns(self):
        if self.receipt is None: return time.perf_counter_ns()
        value,self.receipt=self.receipt,None
        return value


class Tap:
    def __init__(self,transport,plan,role,journal,clock,deadline_ns):
        self.transport,self.plan,self.role=transport,plan,role
        self.journal,self.clock,self.deadline_ns=journal,clock,deadline_ns
        self.dispatch_ns,self.payload=None,None
        self.semantics='fake_transport_entry'

    def sent(self,stamp):
        require(self.dispatch_ns is None,'duplicate HTTP dispatch')
        self.dispatch_ns,self.semantics=stamp,'httpx_request_hook'

    async def stream_completion(self,url,headers,payload,timeout_s,*,request_id=None):
        role,plan=self.role,self.plan
        require(payload['prompt']==plan['prompts'][role] and
                payload['max_tokens']==payload['min_tokens']==plan['outputs'][role] and
                payload['seed']==plan['seeds'][role] and payload['temperature']==0 and
                request_id==plan['request_ids'][role],'wire request differs from plan')
        self.payload=payload
        self.journal.emit('request_wire',role=role,request_id=request_id,payload=payload,
                          scheduled_dispatch_ns=self.deadline_ns)
        hooks=getattr(self.transport,'send_hooks',None)
        if hooks is not None: hooks[request_id]=self.sent
        else: self.dispatch_ns=time.perf_counter_ns()
        stream=self.transport.stream_completion(url,dict(headers,**{'X-Request-ID':request_id}),
                                                payload,timeout_s,request_id=request_id)
        try:
            index=0
            async for raw in stream:
                received=time.perf_counter_ns()
                require(self.dispatch_ns is not None,'missing outbound HTTP timestamp')
                self.journal.emit('sse_wire',role=role,request_id=request_id,event_index=index,
                    raw_data=raw,receive_ns=received,send_ns=self.dispatch_ns)
                self.clock.receipt=received
                yield raw
                index+=1
        finally:
            await stream.aclose()
            if hooks is not None: hooks.pop(request_id,None)


async def shared_httpx_transport(base):
    """Reuse original HttpxTransport methods with one pool and per-call timeouts.

    This avoids constructing SSL/client state separately at each burst dispatch.
    The original executor and its SSE parsing remain unchanged.
    """
    transport=base.HttpxTransport()
    module=transport._httpx
    transport.send_hooks={}
    async def hook(request):
        if request.url.path=='/v1/completions':
            request_id=request.headers.get('X-Request-ID')
            require(request_id in transport.send_hooks,'undeclared HTTP request')
            transport.send_hooks[request_id](time.perf_counter_ns())
    client=module.AsyncClient(trust_env=False,follow_redirects=False,
                              limits=module.Limits(max_connections=16,max_keepalive_connections=16),
                              event_hooks={'request':[hook]})
    class Proxy:
        def __init__(self,timeout): self.timeout=timeout
        async def get(self,url,**kwargs):
            return await client.get(url,timeout=self.timeout,**kwargs)
        def stream(self,*args,**kwargs):
            return client.stream(*args,timeout=self.timeout,**kwargs)
    class Scope:
        def __init__(self,timeout): self.proxy=Proxy(timeout)
        async def __aenter__(self): return self.proxy
        async def __aexit__(self,*args): return False
    transport._httpx=SimpleNamespace(AsyncClient=Scope)
    transport.close=client.aclose
    return transport
