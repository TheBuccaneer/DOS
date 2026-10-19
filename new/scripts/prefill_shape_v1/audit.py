"""Independent episode endpoints and technical gates from raw evidence."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from design import MODEL,SETTINGS,require,sha,validate_episode

ITERATION=re.compile(r'Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, '
                     r'(\d+) generation requests, (\d+) generation tokens, iteration elapsed time: ([0-9.]+) ms')

# vLLM 0.17.1 omits default CLI values from `non-default args`.
# These defaults are checked again against /server_info's resolved configuration.
API_DEFAULTS=dict(cpu_offload_gb=0,gpu_memory_utilization=.9,swap_space=4,
    kv_cache_dtype='auto',kv_cache_memory_bytes=None,kv_offloading_size=None,
    kv_offloading_backend='native',calculate_kv_scales=False,kv_sharing_fast_prefill=False,
    tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,
    data_parallel_size_local=None,data_parallel_rank=None,decode_context_parallel_size=1,
    prefill_context_parallel_size=1,distributed_executor_backend=None,
    enable_expert_parallel=False,enable_dbo=False,disable_custom_all_reduce=False,
    max_num_partial_prefills=1,max_long_partial_prefills=1,long_prefill_token_threshold=0,
    scheduling_policy='fcfs',scheduler_cls=None,async_scheduling=None,
    disable_hybrid_kv_cache_manager=None,disable_chunked_mm_input=False,stream_interval=1,
    disable_sliding_window=False,disable_cascade_attn=False,enforce_eager=False,
    quantization=None,speculative_config=None,kv_transfer_config=None,kv_events_config=None,
    offload_backend='auto',offload_group_size=0,enable_lora=False,compilation_config=None,
    attention_config=None)

CACHE_CONSTANTS=dict(enable_prefix_caching=False,block_size=16,num_gpu_blocks_override=1024,
    cpu_offload_gb=0,gpu_memory_utilization=.9,swap_space=4,cache_dtype='auto',
    kv_cache_memory_bytes=None,kv_offloading_size=None,kv_offloading_backend='native',
    calculate_kv_scales=False,kv_sharing_fast_prefill=False,sliding_window=None,
    is_attention_free=False)


def resolved_configuration(sample,budget):
    require(sample['http_status']==200 and sample['error'] is None,'failed server configuration snapshot')
    require(sample['response_ns']>=sample['scrape_start_ns'],'configuration time reversed')
    body=sample['body']
    require(isinstance(body,dict) and isinstance(body.get('vllm_config'),dict),
            'resolved JSON vllm_config missing')
    config=body['vllm_config']
    groups=dict(
        scheduler_config=dict(runner_type='generate',max_num_batched_tokens=budget,
            # 0.17.1 uses max_num_batched_tokens when this field is None
            # (v1/core/sched/scheduler.py); no speculative reservation applies.
            max_num_scheduled_tokens=None,max_num_seqs=16,max_num_partial_prefills=1,
            max_long_partial_prefills=1,long_prefill_token_threshold=0,enable_chunked_prefill=True,
            policy='fcfs',scheduler_cls=None,async_scheduling=True,
            disable_hybrid_kv_cache_manager=False,disable_chunked_mm_input=False,
            stream_interval=1,is_multimodal_model=False),
        cache_config=CACHE_CONSTANTS,
        parallel_config=dict(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,
            data_parallel_size_local=1,data_parallel_rank=0,decode_context_parallel_size=1,
            prefill_context_parallel_size=1,world_size=1,distributed_executor_backend='uni',
            enable_expert_parallel=False,enable_dbo=False,disable_custom_all_reduce=False),
        model_config=dict(dtype='torch.bfloat16',max_model_len=4096,seed=20261017,
            quantization=None,enforce_eager=False,disable_sliding_window=False,
            disable_cascade_attn=False,generation_config='vllm',served_model_name=MODEL),
        device_config=dict(device='cuda'),
        offload_config=dict(offload_backend='auto'),
        observability_config=dict(enable_logging_iteration_details=True),
        load_config=dict(load_format='auto'),
        compilation_config=dict(mode=3,backend='inductor',cudagraph_mode=[2,1],
            max_cudagraph_capture_size=32,cudagraph_capture_sizes=[1,2,4,8,16,24,32],
            use_inductor_graph_partition=False,cudagraph_num_of_warmups=1),
        attention_config=dict(backend=None,flash_attn_version=None,use_prefill_decode_attention=False,
            use_cudnn_prefill=False,disable_flashinfer_prefill=False,
            use_prefill_query_quantization=False))
    for group,expected in groups.items():
        actual=config.get(group)
        require(isinstance(actual,dict),'missing configuration group: '+group)
        for key,value in expected.items():
            require(key in actual and actual[key]==value and
                    (not isinstance(value,bool) or type(actual[key]) is bool) and
                    (type(value) not in (int,float) or type(actual[key]) in (int,float)),
                    'resolved configuration differs: '+group+'.'+key)
    for key in ('speculative_config','kv_transfer_config','kv_events_config','lora_config'):
        require(key in config and config[key] is None,'resolved configuration differs: '+key)
    offload=config['offload_config']
    for group,key in (('uva','cpu_offload_gb'),('prefetch','offload_group_size')):
        values=offload.get(group)
        require(isinstance(values,dict) and type(values.get(key)) in (int,float) and values[key]==0,
                'resolved CPU offload differs: '+group)
    return config


def gpu_configuration(inventory,environment):
    rows=list(csv.reader(inventory.strip().splitlines(),skipinitialspace=True))
    require(len(rows)==1 and len(rows[0])==5,'single GPU inventory required')
    index,uuid,name,memory,driver=rows[0]
    require(index=='0' and uuid.startswith('GPU-') and name=='NVIDIA GeForce RTX 3090' and
            re.fullmatch(r'24576\s+MiB',memory) and driver.strip(),'RTX 3090 24 GiB on device 0 required')
    require(environment.get('CUDA_VISIBLE_DEVICES')=='0' and
            environment.get('VLLM_SERVER_DEV_MODE')=='1','live server CUDA/configuration environment differs')


def metric(body,name):
    require(isinstance(body,str),'metrics body missing')
    values=[float(m[1]) for m in re.finditer(r'^'+re.escape(name)+r'(?:\{[^\n]*\})?\s+([^\s]+)',body,re.M)]
    require(values and all(math.isfinite(v) for v in values),'missing/nonfinite metric: '+name)
    return sum(values)


def state(sample,budget):
    require(sample['http_status']==200 and sample['error'] is None,'failed metric scrape')
    body=sample['raw_body']
    info=re.findall(r'^vllm:cache_config_info\{([^\n]+)\}\s+1(?:\.0)?$',body,re.M)
    require(len(info)==1,'unique live cache configuration required')
    labels=dict(re.findall(r'(\w+)="([^"\\]*)"',info[0]))
    for key,value in dict(CACHE_CONSTANTS,num_gpu_blocks=1024).items():
        require(key in labels,'effective cache setting missing: '+key)
        actual=labels[key]
        if type(value) in (int,float):
            try: actual=float(actual)
            except ValueError: pass
        else: value=str(value)
        require(actual==value,'effective cache setting differs: '+key)
    counters={name:metric(body,'vllm:'+remote) for name,remote in dict(
        requests='request_success_total',prompt='prompt_tokens_total',generation='generation_tokens_total',
        preemptions='num_preemptions_total',context='request_prefill_kv_computed_tokens_sum').items()}
    return dict(counters=counters,running=metric(body,'vllm:num_requests_running'),
                waiting=metric(body,'vllm:num_requests_waiting'),
                process_start=metric(body,'process_start_time_seconds'))


def process_start(pid):
    ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    boot=next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
    import os
    return boot+ticks/os.sysconf('SC_CLK_TCK')


def startup(raw,budget):
    headers=re.findall(r'^SHAPE_CONFIG_JSON: (.*)$',raw,re.M)
    require(len(headers)==1,'missing unique shape launcher configuration')
    config=json.loads(headers[0])
    require(config==dict(SETTINGS,budget=budget,host='127.0.0.1',port=8018),
            'launcher configuration differs')
    require(re.search(r'^SHAPE_VLLM_VERSION: 0\.17\.1$',raw,re.M),'vLLM version mismatch')
    args_lines=re.findall(r'non-default args: (.*)',raw)
    require(len(args_lines)==1,'unique effective API args required')
    args=ast.literal_eval(args_lines[0])
    require(isinstance(args,dict),'invalid effective API args')
    expected=dict(model=MODEL,max_num_batched_tokens=budget,max_model_len=4096,
                  num_gpu_blocks_override=1024,block_size=16,max_num_seqs=16,
                  enable_prefix_caching=False,enable_chunked_prefill=True,
                  enable_logging_iteration_details=True,seed=20261017)
    require(all(args.get(k)==v for k,v in expected.items()),'effective API scheduler configuration differs')
    for key,value in API_DEFAULTS.items():
        require(args.get(key,value)==value,'effective API constant differs: '+key)
    frontend=dict(host='127.0.0.1',port=8018,dtype='bfloat16',generation_config='vllm',
                  served_model_name=[MODEL],model_tag=MODEL)
    for key,value in frontend.items():
        require(args.get(key)==value,'effective API frontend/model setting differs: '+key)
    require(set(args)<=set(expected)|set(API_DEFAULTS)|set(frontend),
            'unreviewed non-default server argument')
    commands=re.findall(r'^SHAPE_COMMAND_JSON: (.*)$',raw,re.M)
    require(len(commands)==1,'unique server command required')
    from launch_server import command
    recorded=json.loads(commands[0])
    require(isinstance(recorded,list) and recorded[1:]==command(budget)[1:],'server command differs')
    require(re.search(r'Chunked prefill is enabled with max_num_batched_tokens='+str(budget)+r'\.',raw),
            'effective chunked-prefill budget missing')
    require('Asynchronous scheduling is enabled.' in raw and
            'Asynchronous scheduling is disabled.' not in raw,'effective asynchronous scheduling differs')
    backends=re.findall(r'Using (\w+) attention backend out of potential backends:',raw)
    require(backends and set(backends)=={'FLASH_ATTN'},'effective attention backend differs')
    require(re.search(r'Overriding num_gpu_blocks=\d+ with num_gpu_blocks_override=1024\b',raw) and
            'GPU KV cache size: 16,384 tokens' in raw,'effective KV capacity missing')
    engine=re.findall(r'Initializing a V1 LLM engine .*? with config: (.*)',raw)
    require(len(engine)==1,'unique effective engine configuration required')
    for key,value in dict(speculative_config='None',dtype='torch.bfloat16',max_seq_len='4096',
            tensor_parallel_size='1',pipeline_parallel_size='1',data_parallel_size='1',
            disable_custom_all_reduce='False',quantization='None',enforce_eager='False',
            kv_cache_dtype='auto',device_config='cuda',seed='20261017',
            served_model_name=MODEL,enable_prefix_caching='False',enable_chunked_prefill='True').items():
        matches=re.findall(r'(?:^|, )'+key+r'=([^,]+)(?:,|$)',engine[0])
        require(matches==[value],'effective engine constant differs: '+key)
    pids=set(re.findall(r'\(APIServer pid=(\d+)\)',raw))
    declared=re.findall(r'^SHAPE_API_PID: (\d+)$',raw,re.M)
    require(len(pids)==len(declared)==1 and declared[0] in pids,'startup process identity mismatch')
    return int(declared[0])


def token_stream(record,prompt,output):
    require(record['status']=='complete','request incomplete/failed')
    ids,times,sizes=[],[],[]
    done=False
    usage,echo=None,None
    prior=-1
    for event in record['raw_sse_events']:
        stamp=event['receive_perf_counter_ns']
        require(type(stamp) is int and stamp>=prior,'nonmonotone SSE receipts')
        prior=stamp
        raw=event['raw_data']
        require(not done,'data after DONE')
        if raw=='[DONE]': done=True; continue
        if raw=='': continue
        obj=json.loads(raw)
        require(isinstance(obj,dict) and 'error' not in obj,'SSE error payload')
        if obj.get('usage') is not None: usage=obj['usage']
        choices=obj.get('choices') or []
        require(isinstance(choices,list) and len(choices)<=1,'unexpected SSE choices')
        require(not choices or isinstance(choices[0],dict),'invalid SSE choice')
        for candidate in (
            obj.get('prompt_token_ids'),
            choices[0].get('prompt_token_ids') if choices else None
        ):
            if candidate is not None:
                require(isinstance(candidate,list) and all(type(t) is int for t in candidate),
                        'invalid prompt ID echo')
                require(echo is None or echo==candidate,'contradictory prompt ID echo')
                echo=candidate
        batch=(choices[0].get('token_ids') or []) if choices else []
        require(isinstance(batch,list) and all(type(t) is int and t>=0 for t in batch),'invalid output IDs')
        ids.extend(batch);times.extend([stamp]*len(batch))
        if batch: sizes.append(len(batch))
    require(done,'missing DONE')
    require(echo==prompt and ids==record['output_token_ids'],'raw prompt/token copy mismatch')
    require(len(ids)==output and isinstance(usage,dict) and usage.get('completion_tokens')==output and
            usage.get('prompt_tokens')==len(prompt),'usage/output length mismatch')
    require(record['done_received'] and record['finish_reason']=='length','request finish mismatch')
    require(record['send_ns']>=record['scheduled_dispatch_ns'] and record['send_ns']<=times[0],
            'invalid send/receive timestamps')
    return dict(ids=ids,times=times,batch_sizes=sizes)


def assess(plan,episode):
    issues=[]
    streams={}
    def check(ok,message):
        if not ok: issues.append(message)
    try: validate_episode(plan)
    except (ValueError,KeyError,TypeError) as exc: issues.append(str(exc))
    try:
        startup(episode['runtime']['startup_log'],plan['budget'])
        if episode['origin']=='serving':
            gpu_configuration(episode['runtime']['gpu_inventory'],episode['runtime']['server_environment'])
        before=episode['server_config_before'];after=episode['server_config_after']
        a=resolved_configuration(before,plan['budget'])
        b=resolved_configuration(after,plan['budget'])
        require(a==b,'resolved server configuration changed during episode')
        require(before['response_ns']<episode['victim_start_ns'],'configuration did not precede victims')
        terminal=max((r.get('stream_end_ns',0) for r in episode['requests'].values()),default=0)
        require(after['scrape_start_ns']>terminal,'configuration did not follow requests')
    except (ValueError,KeyError,TypeError) as exc: issues.append('server configuration: '+str(exc))
    check(not episode['errors'],'episode errors present')
    check(set(episode['requests'])==set(plan['prompts']),'request cohort incomplete/unexpected')
    for role in plan['prompts']:
        try:
            record=episode['requests'][role]
            require(record['request_id']==plan['request_ids'][role],'request identity mismatch')
            require(record['wire_payload']==dict(model=MODEL,prompt=plan['prompts'][role],
                    max_tokens=plan['outputs'][role],min_tokens=plan['outputs'][role],ignore_eos=True,
                    temperature=0,stream=True,stream_options=dict(include_usage=True),
                    add_special_tokens=False,return_token_ids=True,seed=plan['seeds'][role]),'wire payload mismatch')
            require(record['victim_phase_start_ns']==episode['victim_start_ns'],'common phase origin changed')
            streams[role]=token_stream(record,plan['prompts'][role],plan['outputs'][role])
            expected=episode['victim_start_ns']+(0 if role.startswith('V') else
                     (plan['trigger_ms']+(int(role[1:])*plan['interval_ms'] if plan['condition']=='distributed' else 0))*1_000_000)
            require(record['scheduled_dispatch_ns']==expected,'request deadline changed')
            late=(record['send_ns']-expected)/1e6
            check(late<=SETTINGS['max_dispatch_lateness_ms'],'dispatch lateness exceeded: '+role)
            if episode['origin']=='serving':
                check(record['dispatch_semantics']=='httpx_request_hook','serving send hook missing: '+role)
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            issues.append(role+': '+str(exc))
    victims=[f'V{i}' for i in range(4)]
    endpoints={v:(streams[v]['times'][63] if v in streams else None) for v in victims}
    raw_primary=((max(endpoints.values())-episode['victim_start_ns'])/1e6
                 if all(v is not None for v in endpoints.values()) else None)
    trigger=episode.get('trigger')
    decoding=False
    if trigger is not None:
        expected=episode['victim_start_ns']+plan['trigger_ms']*1_000_000
        check(trigger['deadline_ns']==expected and trigger['fired_ns']>=expected,'timer fired early/wrong deadline')
        check((trigger['fired_ns']-expected)/1e6<=SETTINGS['max_dispatch_lateness_ms'],'trigger late')
        decoding=all(v in streams and streams[v]['times'][0]<trigger['fired_ns']<streams[v]['times'][63]
                     for v in victims)
    else: issues.append('missing timer trigger')
    scheduler={}
    coverage={}
    try:
        samples=episode['metrics_samples']
        require(len(samples)>=2,'missing before/after metrics')
        require(episode['metrics_before']==samples[0] and episode['metrics_after']==samples[-1],
                'metric boundary copy mismatch')
        states=[state(s,plan['budget']) for s in samples]
        require(all(s['response_ns']>=s['scrape_start_ns'] for s in samples),'metrics time reversed')
        a,b=states[0],states[-1]
        require(a['running']==a['waiting']==b['running']==b['waiting']==0,'nonidle metric boundaries')
        require(all(abs(s['process_start']-episode['runtime']['process_start'])<2 for s in states),
                'metrics belong to another server process')
        require(samples[0]['response_ns']<episode['victim_start_ns'],'metrics did not precede victims')
        terminal=max(r['stream_end_ns'] for r in episode['requests'].values() if 'stream_end_ns' in r)
        require(samples[-1]['scrape_start_ns']>max(terminal,trigger['fired_ns'] if trigger else 0),
                'metrics did not follow all requests/timer')
        gaps=[(y['scrape_start_ns']-x['scrape_start_ns'])/1e6 for x,y in zip(samples,samples[1:])]
        require(all(0<=g<=SETTINGS['max_metrics_gap_ms'] for g in gaps),'metrics coverage gap')
        coverage=dict(samples=len(samples),max_gap_ms=max(gaps),continuous=True)
        expected=dict(requests=len(plan['prompts']),prompt=sum(map(len,plan['prompts'].values())),
                      generation=sum(plan['outputs'].values()),preemptions=0,
                      context=sum(map(len,plan['prompts'].values())))
        for key,value in expected.items():
            require(all(y['counters'][key]>=x['counters'][key] for x,y in zip(states,states[1:])),
                    'counter reset: '+key)
            require(b['counters'][key]-a['counters'][key]==value,'counter/foreign-load mismatch: '+key)
        raw=episode['iteration_log']
        require(hashlib.sha256(raw.encode()).hexdigest()==episode['iteration_log_sha256'],'iteration log hash mismatch')
        records=[]
        for line in raw.splitlines():
            if 'Iteration(' not in line: continue
            m=ITERATION.search(line)
            require(m is not None,'malformed scheduler iteration')
            records.append(tuple(map(int,m.groups()[:5])))
        require(records and all(b[0]==a[0]+1 for a,b in zip(records,records[1:])),
                'missing/gapped scheduler trajectory')
        require(sum(r[2] for r in records)==expected['context'],'scheduler context accounting mismatch')
        peak=max(r[2]+r[4] for r in records)
        require(peak<=plan['budget'],'scheduler budget exceeded')
        scheduler=dict(iterations=len(records),max_scheduled_tokens=peak,
                       budget_binding_observed=peak>=plan['budget']-8,
                       prefill_decode_overlap_iterations=sum(r[1]>0 and r[3]>0 for r in records),
                       gpu_kernel_time_measured=False,request_attribution='aggregate_only')
    except (ValueError,KeyError,TypeError,IndexError) as exc:
        issues.append(str(exc))
    extra=[episode['requests'][r]['send_ns'] for r in plan['prompts'] if r.startswith('E') and
           episode['requests'].get(r,{}).get('send_ns') is not None]
    victim_sends=[episode['requests'][v].get('send_ns') for v in victims if v in episode['requests']]
    victim_sends=[s for s in victim_sends if s is not None]
    return dict(schema='prefill-shape-assessment-v1',mode=plan['mode'],origin=episode['origin'],
                primary_ms=raw_primary,victim_token64_receipts_ns=endpoints,
                measurement_valid=not issues,issues=issues,
                all_victims_decoding_at_trigger=decoding,
                eligible_for_planned_comparison=not issues and decoding,
                extra_dispatch_span_ms=(max(extra)-min(extra))/1e6 if extra else None,
                extra_dispatch_gaps_ms=[(y-x)/1e6 for x,y in zip(extra,extra[1:])],
                victim_dispatch_span_ms=(max(victim_sends)-min(victim_sends))/1e6 if victim_sends else None,
                victims_decoding_at_each_extra=[sum(v in streams and streams[v]['times'][0]<stamp<streams[v]['times'][63]
                                                    for v in victims) for stamp in extra],
                trigger_lateness_ms=(trigger['fired_ns']-trigger['deadline_ns'])/1e6 if trigger else None,
                metrics_coverage=coverage,scheduler=scheduler,scientific_result_claimed=False)


def read_episode(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    require(rows and rows[0]['record_type']=='episode_start' and
            rows[-1]['record_type']=='episode_result','incomplete journal; keep it as incomplete')
    require([r['journal_seq'] for r in rows]==list(range(len(rows))),'journal sequence gap')
    starts=[r for r in rows if r['record_type']=='episode_start']
    finals=[r for r in rows if r['record_type']=='episode_result']
    require(len(starts)==len(finals)==1,'duplicate episode lifecycle')
    plan=starts[0]['plan'];episode=finals[0]['episode']
    validate_episode(plan)
    allowed={'episode_start','victim_start','metrics_sample','request_wire','sse_wire',
             'request_result','timer_trigger','iteration_log','episode_result','server_config'}
    require(all(r['record_type'] in allowed for r in rows),'undeclared journal event')
    require(starts[0]['plan_sha256']==sha(plan) and episode['plan']==plan,'plan/journal mismatch')
    require(starts[0]['runtime']==episode['runtime'] and starts[0]['origin']==episode['origin'],
            'startup/runtime/origin copy changed')
    require([r['victim_start_ns'] for r in rows if r['record_type']=='victim_start']==
            ([] if episode['victim_start_ns'] is None else [episode['victim_start_ns']]),'victim origin journal mismatch')
    runtime=episode['runtime']
    require(hashlib.sha256(runtime['startup_log'].encode()).hexdigest()==runtime['startup_sha256'],
            'startup log hash mismatch')
    require(startup(runtime['startup_log'],plan['budget'])==runtime['api_pid'],'startup API PID mismatch')
    samples=[r['sample'] for r in rows if r['record_type']=='metrics_sample']
    require(samples==episode['metrics_samples'],'metrics differ from raw journal')
    for phase in ('before','after'):
        require([r['sample'] for r in rows if r['record_type']=='server_config' and r['phase']==phase]==
                [episode['server_config_'+phase]],'server configuration journal/copy mismatch')
    require(all(r['phase'] in ('before','after') for r in rows if r['record_type']=='server_config'),
            'unknown server configuration phase')
    require([r['trigger'] for r in rows if r['record_type']=='timer_trigger']==
            ([] if episode['trigger'] is None else [episode['trigger']]),'timer journal mismatch')
    logs=[r for r in rows if r['record_type']=='iteration_log']
    if 'log_byte_end' in episode:
        require(len(logs)==1 and logs[0]['raw_text']==episode['iteration_log'] and
                logs[0]['byte_start']==episode['log_byte_start'] and logs[0]['byte_end']==episode['log_byte_end'] and
                episode['log_byte_end']-episode['log_byte_start']==len(episode['iteration_log'].encode()),
                'iteration journal/copy mismatch')
    for role,record in episode['requests'].items():
        wire=[r for r in rows if r['record_type']=='sse_wire' and r['role']==role]
        require([(r['raw_data'],r['receive_ns']) for r in wire]==
                [(r['raw_data'],r['receive_perf_counter_ns']) for r in record.get('raw_sse_events',[])],
                'SSE journal copy mismatch')
        require([r['event_index'] for r in wire]==list(range(len(wire))) and
                all(r['request_id']==plan['request_ids'][role] and r['send_ns']==record['send_ns'] for r in wire),
                'wire identity/timestamp mismatch')
        declared=[r for r in rows if r['record_type']=='request_wire' and r['role']==role]
        payloads=[r['payload'] for r in declared]
        copies=[r['record'] for r in rows if r['record_type']=='request_result' and r['role']==role]
        if record.get('status')=='not_dispatched':
            require(not payloads and not copies and not wire,'unexpected request')
        else:
            require(payloads==[record['wire_payload']] and copies==[record],'request result differs from journal')
            require(declared[0]['request_id']==plan['request_ids'][role] and
                    declared[0]['scheduled_dispatch_ns']==record['scheduled_dispatch_ns'],'wire deadline/identity changed')
    require(all(r.get('role') in plan['prompts'] for r in rows if r['record_type'] in
                ('request_wire','request_result','sse_wire')),'undeclared request role')
    assessment=assess(plan,episode)
    require(assessment==finals[0]['assessment'],'stored assessment differs from independent reconstruction')
    return assessment
