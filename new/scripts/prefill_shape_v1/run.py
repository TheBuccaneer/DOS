#!/usr/bin/env python3
"""Prepare plans, run exactly one triplet, or independently audit episodes."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

from design import BASE_SHA256,CONDITIONS,MODEL,SETTINGS,load_base,prepare,require,sha,validate
from instrument import Journal,ReceiptClock,Sampler,Tap,local_url,shared_httpx_transport,sleep_until
from audit import assess,read_episode,state,startup,process_start,resolved_configuration,gpu_configuration


def runtime_evidence(server_log,budget):
    require(importlib.metadata.version('vllm')=='0.17.1','vLLM 0.17.1 runtime required')
    raw=server_log.read_text()
    pid=startup(raw,budget)
    inventory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version',
                              '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    require(inventory.returncode==0,'GPU inventory unavailable')
    environment=dict(part.split('=',1) for part in Path(f'/proc/{pid}/environ').read_text().split('\0') if '=' in part)
    environment={key:environment.get(key) for key in ('CUDA_VISIBLE_DEVICES','VLLM_SERVER_DEV_MODE')}
    gpu_configuration(inventory.stdout,environment)
    return dict(startup_log=raw,startup_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                api_pid=pid,process_start=process_start(pid),gpu_inventory=inventory.stdout.strip(),
                server_environment=environment,
                clock='time.perf_counter_ns',boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                historical_executor_sha256=BASE_SHA256)


async def run_episode(base,transport,plan,journal,runtime,server_log,*,base_url,api_key,origin='serving'):
    episode=dict(plan=plan,runtime=runtime,origin=origin,requests={},trigger=None,errors=[],
                 victim_start_ns=None,metrics_samples=[],metrics_before=None,metrics_after=None,
                 iteration_log='',iteration_log_sha256=hashlib.sha256(b'').hexdigest(),
                 server_config_before=None,server_config_after=None)
    journal.emit('episode_start',plan=plan,plan_sha256=sha(plan),runtime=runtime,origin=origin)
    sampler=Sampler(transport,base_url+'/metrics',journal,api_key)
    tasks=[]
    log_stat=None
    cancelled=False

    async def configuration_snapshot(phase):
        sample=dict(scrape_start_ns=time.perf_counter_ns(),http_status=None,body=None,error=None)
        try:
            status,body=await transport.get_json(base_url+'/server_info?config_format=json',
                {'Authorization':'Bearer '+api_key},10)
            require(api_key not in json.dumps(body),'server configuration contains authentication secret')
            sample.update(http_status=status,body=body)
            if status!=200: sample['error']='http_error'
        except Exception as exc: sample['error']=type(exc).__name__
        sample['response_ns']=time.perf_counter_ns()
        episode['server_config_'+phase]=sample
        journal.emit('server_config',phase=phase,sample=sample)
        return sample

    async def execute(role,deadline):
        clock=ReceiptClock(base.RealClock())
        tap=Tap(transport,plan,role,journal,clock,deadline)
        created=time.perf_counter_ns()
        record=await base.execute_completion_request(
            transport=tap,clock=clock,url=base_url+'/v1/completions',api_key=api_key,
            model_full_id=MODEL,prompt_token_ids=plan['prompts'][role],max_tokens=plan['outputs'][role],
            min_tokens=plan['outputs'][role],temperature=0,request_seed=plan['seeds'][role],
            request_id=plan['request_ids'][role],role='victim' if role.startswith('V') else 'additional',
            request_index=int(role[1:]),prompt_seed=plan['seeds'][role],generation_seed=plan['seeds'][role],
            expected_prompt_tokens=len(plan['prompts'][role]),expected_completion_tokens=plan['outputs'][role],
            http_timeout_s=SETTINGS['request_timeout_s'],task_created_ns=created,
            victim_phase_start_ns=episode['victim_start_ns'])
        record.update(send_ns=tap.dispatch_ns,dispatch_semantics=tap.semantics,
                      scheduled_dispatch_ns=deadline,wire_payload=tap.payload)
        episode['requests'][role]=record
        journal.emit('request_result',role=role,record=record)
        return record

    async def timer():
        # This function never reads victim progress, completion or output tokens.
        deadline=episode['victim_start_ns']+plan['trigger_ms']*1_000_000
        await sleep_until(deadline)
        episode['trigger']=dict(deadline_ns=deadline,fired_ns=time.perf_counter_ns(),
                                 server_log_offset=server_log.stat().st_size)
        journal.emit('timer_trigger',trigger=episode['trigger'])
        if plan['condition']=='control': return
        for i in range(4):
            scheduled=deadline+(i*plan['interval_ms']*1_000_000 if plan['condition']=='distributed' else 0)
            await sleep_until(scheduled)
            tasks.append(asyncio.create_task(execute(f'E{i}',scheduled)))

    async def workload():
        episode['victim_start_ns']=time.perf_counter_ns()
        journal.emit('victim_start',victim_start_ns=episode['victim_start_ns'])
        for i in range(4):
            tasks.append(asyncio.create_task(execute(f'V{i}',episode['victim_start_ns'])))
        trigger_task=asyncio.create_task(timer())
        tasks.append(trigger_task)
        await trigger_task
        await asyncio.gather(*tasks)

    try:
        resolved_configuration(await configuration_snapshot('before'),plan['budget'])
        first=await sampler.sample()
        initial=state(first,plan['budget'])
        require(initial['running']==initial['waiting']==0,'server not idle before episode')
        require(abs(initial['process_start']-runtime['process_start'])<2,'startup log not bound to live server')
        log_stat=server_log.stat()
        episode['log_byte_start']=log_stat.st_size
        sampler.start()
        await asyncio.wait_for(workload(),SETTINGS['episode_timeout_s'])
    except asyncio.CancelledError:
        cancelled=True
        episode['errors'].append(dict(type='CancelledError',detail='episode interrupted'))
    except Exception as exc:
        episode['errors'].append(dict(type=type(exc).__name__,detail=str(exc).replace(api_key,'<redacted>')[:500]))
    finally:
        await base.cancel_all(tasks)
        for role in plan['prompts']:
            episode['requests'].setdefault(role,dict(request_id=plan['request_ids'][role],status='not_dispatched'))
        try:
            terminal=max((r.get('stream_end_ns',0) for r in episode['requests'].values()),default=0)
            deadline=time.monotonic()+5
            if sampler.task is not None:
                while time.monotonic()<deadline:
                    last=sampler.samples[-1]
                    try:
                        latest=state(last,plan['budget'])
                        expected=sum(r.get('status')=='complete' for r in episode['requests'].values())
                        if (last['scrape_start_ns']>terminal and latest['running']==latest['waiting']==0 and
                            latest['counters']['requests']-initial['counters']['requests']>=expected): break
                    except (ValueError,KeyError): pass
                    await asyncio.sleep(.05)
                await asyncio.sleep(.15)  # fixed log flush allowance; sampler keeps running
            await sampler.stop()
            await sampler.sample()  # full AFTER sample; no concurrent scrape remains
        except Exception as exc:
            episode['errors'].append(dict(type=type(exc).__name__,detail='metric finalization failed'))
        await configuration_snapshot('after')
        episode['metrics_samples']=sampler.samples
        episode['metrics_before']=sampler.samples[0] if sampler.samples else None
        episode['metrics_after']=sampler.samples[-1] if sampler.samples else None
        if log_stat is not None:
            try:
                last=server_log.stat()
                require((last.st_dev,last.st_ino)==(log_stat.st_dev,log_stat.st_ino) and
                        last.st_size>=log_stat.st_size,'server log replaced/truncated')
                with server_log.open('rb') as stream:
                    stream.seek(log_stat.st_size)
                    segment=stream.read()
                episode.update(iteration_log=segment.decode(),iteration_log_sha256=hashlib.sha256(segment).hexdigest(),
                               log_byte_end=log_stat.st_size+len(segment))
                journal.emit('iteration_log',raw_text=episode['iteration_log'],
                              byte_start=episode['log_byte_start'],byte_end=episode['log_byte_end'])
            except Exception as exc:
                episode['errors'].append(dict(type=type(exc).__name__,detail='iteration log unavailable'))
        result=assess(plan,episode)
        journal.emit('episode_result',episode=episode,assessment=result)
        await journal.flush()
    if cancelled: raise asyncio.CancelledError
    return result


async def run_block(plan,budget,block,server_log,output_dir,base_url,api_key):
    episodes=[e for e in plan['episodes'] if e['budget']==budget and e['paired_block']==block]
    require(len(episodes)==3,'select exactly one declared triplet')
    output_dir.mkdir(parents=True,exist_ok=False)
    summary=[]
    transport=None
    failure=None
    try:
        runtime=runtime_evidence(server_log,budget)
        require(api_key not in runtime['startup_log'],'startup log contains authentication secret')
        base=load_base()
        transport=await shared_httpx_transport(base)
        for ep in episodes:
            path=output_dir/(ep['episode_id']+'.jsonl')
            journal=Journal(path)
            try:
                result=await run_episode(base,transport,ep,journal,runtime,server_log,
                                         base_url=base_url,api_key=api_key)
            finally: journal.close()
            checked=read_episode(path)
            require(checked==result,'independent assessment mismatch')
            summary.append(dict(episode_id=ep['episode_id'],path=path.name,assessment=result))
    except (Exception,asyncio.CancelledError) as exc:
        failure=dict(type=type(exc).__name__,detail=str(exc).replace(api_key,'<redacted>')[:500])
    finally:
        if transport is not None: await transport.close()
        report=dict(mode=plan['mode'],budget=budget,block=block,episodes=summary,
                    error=failure,plan_sha256=sha(plan),
                    planned_episode_ids=[e['episode_id'] for e in episodes],
                    not_completed=[e['episode_id'] for e in episodes if e['episode_id'] not in
                                   {r['episode_id'] for r in summary}],scientific_result_claimed=False)
        with (output_dir/'block_summary.json').open('x') as f:
            json.dump(report,f,indent=2)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    q.add_argument('--mode',choices=('diagnostic','main'),default='diagnostic')
    q.add_argument('--budget',type=int,choices=(512,2048),
                   help='diagnostic budget only; defaults to 2048')
    q.add_argument('--trigger-ms',type=int)
    q.add_argument('--interval-ms',type=int)
    q.add_argument('--token-ids',type=Path)
    q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('run-block')
    q.add_argument('--plan',type=Path,required=True)
    q.add_argument('--budget',type=int,choices=(512,2048),required=True)
    q.add_argument('--block',type=int,required=True)
    q.add_argument('--server-log',type=Path,required=True)
    q.add_argument('--output-dir',type=Path,required=True)
    q.add_argument('--base-url',default='http://127.0.0.1:8018')
    q.add_argument('--allow-main-block',action='store_true')
    q=sub.add_parser('audit')
    q.add_argument('episodes',type=Path,nargs='+')
    a=p.parse_args()
    try:
        if a.command=='prepare':
            require(a.mode=='diagnostic' or a.budget is None,'--budget selects diagnostic plans only')
            plan=prepare(a.mode,trigger_ms=a.trigger_ms,interval_ms=a.interval_ms,
                         tokens=json.loads(a.token_ids.read_text()) if a.token_ids else None,
                         diagnostic_budget=a.budget if a.budget is not None else 2048)
            with a.output.open('x') as f: json.dump(plan,f,indent=2)
            print(json.dumps(dict(mode=plan['mode'],episodes=len(plan['episodes']),
                                 completion_requests=plan['completion_request_count'],plan_sha256=sha(plan))))
        elif a.command=='audit':
            reports=[]
            for path in a.episodes:
                try: reports.append(dict(path=str(path),assessment=read_episode(path)))
                except (ValueError,OSError,KeyError,TypeError) as exc:
                    reports.append(dict(path=str(path),status='incomplete_or_invalid',reason=str(exc)))
            print(json.dumps(reports,indent=2))
            return int(any(not r.get('assessment',{}).get('measurement_valid') for r in reports))
        else:
            plan=validate(json.loads(a.plan.read_text()))
            require(plan['mode']=='diagnostic' or a.allow_main_block,'main block requires explicit --allow-main-block')
            base_url=local_url(a.base_url)
            require(base_url=='http://127.0.0.1:8018','fixed instrumented port is 8018')
            api_key=os.environ.get('VLLM_API_KEY')
            require(bool(api_key),'VLLM_API_KEY required; value is never recorded')
            report=asyncio.run(run_block(plan,a.budget,a.block,a.server_log,a.output_dir,base_url,api_key))
            print(json.dumps(report,indent=2))
            return int(bool(report['not_completed']) or bool(report['error']) or
                       any(not e['assessment']['eligible_for_planned_comparison'] for e in report['episodes']))
    except (ValueError,OSError,KeyError,TypeError,importlib.metadata.PackageNotFoundError) as exc:
        p.exit(1,type(exc).__name__+': '+str(exc)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
