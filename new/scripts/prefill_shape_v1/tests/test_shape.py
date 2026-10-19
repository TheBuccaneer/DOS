"""Offline checks: real frozen executor, fake streams, no GPU or network."""
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from design import CONDITIONS,MODEL,SETTINGS,load_base,prepare,validate,sha
from instrument import Journal,shared_httpx_transport,sleep_until,local_url
from audit import assess,read_episode,startup,token_stream,state,resolved_configuration,gpu_configuration
from run import run_episode,run_block,runtime_evidence
from launch_server import command


def startup_text(budget):
    args=dict(model=MODEL,max_num_batched_tokens=budget,max_model_len=4096,
              num_gpu_blocks_override=1024,block_size=16,max_num_seqs=16,
              enable_prefix_caching=False,enable_chunked_prefill=True,
              enable_logging_iteration_details=True,seed=20261017,model_tag=MODEL,
              host='127.0.0.1',port=8018,dtype='bfloat16',generation_config='vllm',served_model_name=[MODEL])
    return ('SHAPE_CONFIG_JSON: '+json.dumps(dict(SETTINGS,budget=budget,host='127.0.0.1',port=8018))+'\n'
        'SHAPE_COMMAND_JSON: '+json.dumps(command(budget))+'\n'
        'SHAPE_VLLM_VERSION: 0.17.1\nSHAPE_API_PID: 4321\n'
        f'(APIServer pid=4321) non-default args: {args!r}\n'
        f'Chunked prefill is enabled with max_num_batched_tokens={budget}.\n'
        'Asynchronous scheduling is enabled.\n'
        "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'TRITON_ATTN'].\n"
        'Overriding num_gpu_blocks=3000 with num_gpu_blocks_override=1024\n'
        'GPU KV cache size: 16,384 tokens\n'
        "Initializing a V1 LLM engine (v0.17.1) with config: model='public-fixture', "
        'speculative_config=None, dtype=torch.bfloat16, max_seq_len=4096, tensor_parallel_size=1, '
        'pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, '
        'quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, '
        f'seed=20261017, served_model_name={MODEL}, enable_prefix_caching=False, enable_chunked_prefill=True\n')


def server_info(budget):
    return dict(vllm_config=dict(
        scheduler_config=dict(runner_type='generate',max_num_batched_tokens=budget,
            max_num_scheduled_tokens=None,max_num_seqs=16,max_num_partial_prefills=1,
            max_long_partial_prefills=1,long_prefill_token_threshold=0,enable_chunked_prefill=True,
            policy='fcfs',scheduler_cls=None,async_scheduling=True,disable_hybrid_kv_cache_manager=False,
            disable_chunked_mm_input=False,stream_interval=1,is_multimodal_model=False),
        cache_config=dict(enable_prefix_caching=False,block_size=16,num_gpu_blocks_override=1024,
            cpu_offload_gb=0,gpu_memory_utilization=.9,swap_space=4,cache_dtype='auto',
            kv_cache_memory_bytes=None,kv_offloading_size=None,kv_offloading_backend='native',
            calculate_kv_scales=False,kv_sharing_fast_prefill=False,sliding_window=None,is_attention_free=False),
        parallel_config=dict(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,
            data_parallel_size_local=1,data_parallel_rank=0,decode_context_parallel_size=1,
            prefill_context_parallel_size=1,world_size=1,distributed_executor_backend='uni',
            enable_expert_parallel=False,enable_dbo=False,disable_custom_all_reduce=False),
        model_config=dict(dtype='torch.bfloat16',max_model_len=4096,seed=20261017,quantization=None,
            enforce_eager=False,disable_sliding_window=False,disable_cascade_attn=False,
            generation_config='vllm',served_model_name=MODEL),
        device_config=dict(device='cuda'),
        offload_config=dict(offload_backend='auto',uva=dict(cpu_offload_gb=0),prefetch=dict(offload_group_size=0)),
        observability_config=dict(enable_logging_iteration_details=True),
        load_config=dict(load_format='auto'),
        compilation_config=dict(mode=3,backend='inductor',cudagraph_mode=[2,1],
            max_cudagraph_capture_size=32,cudagraph_capture_sizes=[1,2,4,8,16,24,32],
            use_inductor_graph_partition=False,cudagraph_num_of_warmups=1),
        attention_config=dict(backend=None,flash_attn_version=None,use_prefill_decode_attention=False,
            use_cudnn_prefill=False,disable_flashinfer_prefill=False,use_prefill_query_quantization=False),
        speculative_config=None,kv_transfer_config=None,kv_events_config=None,lora_config=None))


def metrics(counters,running):
    labels=('enable_prefix_caching="False",block_size="16",num_gpu_blocks="1024",'
            'num_gpu_blocks_override="1024",cpu_offload_gb="0",gpu_memory_utilization="0.9",'
            'swap_space="4.0",cache_dtype="auto",kv_cache_memory_bytes="None",kv_offloading_size="None",'
            'kv_offloading_backend="native",calculate_kv_scales="False",kv_sharing_fast_prefill="False",'
            'sliding_window="None",is_attention_free="False"')
    names=dict(requests='request_success_total',prompt='prompt_tokens_total',generation='generation_tokens_total',
               preemptions='num_preemptions_total',context='request_prefill_kv_computed_tokens_sum')
    return '\n'.join([f'vllm:cache_config_info{{{labels}}} 1','process_start_time_seconds 1000.25',
                      f'vllm:num_requests_running {running}','vllm:num_requests_waiting 0']+
                     [f'vllm:{remote} {counters[name]}' for name,remote in names.items()])


class FakeTransport:
    def __init__(self,plan,log,fault=None):
        self.plan,self.log,self.fault=plan,log,fault
        self.active=set()
        self.sent=[]
        self.index,self.scrapes,self.max_active=0,0,0
        self.configuration_scrapes=0
        self.counters=Counter()

    async def get_json(self,url,headers,timeout_s):
        if url.endswith('/server_info?config_format=json'):
            self.configuration_scrapes+=1
            if self.fault=='config_http': return 404,{'detail':'not enabled'}
            body=server_info(self.plan['budget'])
            if self.fault=='config_before' or (self.fault=='config_after' and self.configuration_scrapes>1):
                body['vllm_config']['scheduler_config']['max_num_partial_prefills']=2
            return 200,body
        self.scrapes+=1
        if self.fault=='gap' and self.scrapes==3: await asyncio.sleep(.3)
        if self.fault=='metrics_http' and self.scrapes==3: return 503,'unavailable'
        await asyncio.sleep(0)
        body=metrics(self.counters,len(self.active))
        if self.fault=='wrong_server': body=body.replace('1000.25','2000.25')
        return 200,body

    def context(self,role,count):
        if self.fault=='rotation' and role=='E0':
            self.log.rename(self.log.with_suffix('.old'))
            self.log.write_text('replacement log\n')
        generation=sum(r.startswith('V') for r in self.active if r!=role) if role.startswith('E') else 0
        with self.log.open('a') as f:
            while count:
                chunk=min(count,self.plan['budget']-generation)
                f.write(f'Iteration({self.index}): 1 context requests, {chunk} context tokens, '
                        f'{generation} generation requests, {generation} generation tokens, '
                        'iteration elapsed time: 1.00 ms\n')
                self.index+=1
                count-=chunk

    async def stream_completion(self,url,headers,payload,timeout_s,*,request_id=None):
        role=request_id.rsplit(':',1)[-1]
        self.sent.append((role,time.perf_counter_ns(),copy.deepcopy(payload)))
        self.active.add(role)
        self.max_active=max(self.max_active,len(self.active))
        self.context(role,len(payload['prompt']))
        count=0
        try:
            echo=dict(prompt_token_ids=payload['prompt'],choices=[dict(index=0,token_ids=[])])
            if self.fault=='nested_echo': echo['choices'][0]['prompt_token_ids']=echo.pop('prompt_token_ids')
            yield json.dumps(echo)
            if self.fault=='timeout' and role=='V1': raise asyncio.TimeoutError('offline timeout')
            for i in range(payload['max_tokens']):
                await asyncio.sleep((.00001 if self.fault=='early' else .02) if role.startswith('V') else .001)
                count+=1
                yield json.dumps(dict(choices=[dict(index=0,token_ids=[100+i],text='')]))
            usage=dict(prompt_tokens=len(payload['prompt']),completion_tokens=count)
            yield json.dumps(dict(choices=[dict(index=0,token_ids=[],finish_reason='length')],usage=usage))
            if not (self.fault=='missing_done' and role=='V1'): yield '[DONE]'
        finally:
            self.active.remove(role)
            self.counters.update(prompt=len(payload['prompt']),context=len(payload['prompt']),generation=count,
                                 requests=int(count==payload['max_tokens']))


def execute(plan,fault=None):
    base=load_base()
    temp=tempfile.TemporaryDirectory()
    path=Path(temp.name)
    log=path/'server.log'
    raw=startup_text(plan['budget'])
    log.write_text(raw)
    runtime=dict(startup_log=raw,startup_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                 api_pid=4321,process_start=1000.25,clock='time.perf_counter_ns',boot_id='offline')
    transport=FakeTransport(plan,log,fault)
    out=path/'episode.jsonl'
    journal=Journal(out)
    try:
        with patch('socket.socket.connect',side_effect=AssertionError('offline network forbidden')):
            result=asyncio.run(run_episode(base,transport,plan,journal,runtime,log,
                base_url='http://127.0.0.1:8018',api_key='offline-secret',origin='synthetic'))
    finally: journal.close()
    rows=[json.loads(s) for s in out.read_text().splitlines()]
    return temp,out,result,rows,transport


class DesignTests(unittest.TestCase):
    def test_diagnostic_512_is_one_triplet_with_paired_2048_workload_and_rules(self):
        small=validate(prepare(study_id='paired',diagnostic_budget=512))
        large=validate(prepare(study_id='paired'))
        self.assertEqual(small['mode'],'diagnostic')
        self.assertEqual(small['budgets'],[512])
        self.assertEqual(small['blocks_per_budget'],1)
        self.assertEqual(small['completion_request_count'],20)
        self.assertEqual(len(small['episodes']),3)
        self.assertEqual({e['condition'] for e in small['episodes']},set(CONDITIONS))
        for ep in small['episodes']:
            paired=next(e for e in large['episodes'] if e['condition']==ep['condition'])
            for key in ('prompts','seeds','outputs','trigger_ms','interval_ms','mode'):
                self.assertEqual(ep[key],paired[key])
        for budget in (0,1024,True):
            with self.assertRaises(ValueError): prepare(diagnostic_budget=budget)
        broken=copy.deepcopy(small);broken['episodes'][0]['budget']=2048
        with self.assertRaises(ValueError): validate(broken)

    def test_cli_prepares_512_without_main_mode_or_main_parameter_freeze(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'diagnostic.json'
            process=subprocess.run([sys.executable,str(HERE/'run.py'),'prepare','--mode','diagnostic',
                '--budget','512','--output',str(path)],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            plan=validate(json.loads(path.read_text()))
            self.assertEqual(plan['budgets'],[512])
            self.assertTrue(all(e['trigger_ms']==800 and e['interval_ms']==100 for e in plan['episodes']))
            process=subprocess.run([sys.executable,str(HERE/'run.py'),'prepare','--mode','main',
                '--budget','512','--output',str(Path(t)/'main.json')],capture_output=True,text=True)
            self.assertNotEqual(process.returncode,0)
            self.assertIn('diagnostic plans only',process.stderr)

    def test_diagnostic_is_exactly_one_budget_2048_triplet(self):
        plan=prepare(study_id='offline')
        self.assertEqual(plan['budgets'],[2048])
        self.assertEqual(plan['blocks_per_budget'],1)
        self.assertEqual(len(plan['episodes']),3)
        self.assertEqual({e['condition'] for e in plan['episodes']},set(CONDITIONS))
        self.assertEqual(plan['completion_request_count'],20)
        self.assertTrue(all(e['trigger_ms']==800 and e['interval_ms']==100 for e in plan['episodes']))

    def test_main_requires_its_own_explicit_timer_freeze(self):
        with self.assertRaisesRegex(ValueError,'separately frozen'): prepare('main')
        with self.assertRaisesRegex(ValueError,'fixed'): prepare(trigger_ms=801)

    def test_main_has_ten_independent_triplets_per_budget_with_paired_workloads(self):
        plan=prepare('main',study_id='main',trigger_ms=800,interval_ms=100)
        self.assertEqual(len(plan['episodes']),60)
        self.assertEqual(plan['completion_request_count'],400)
        orders=set()
        for block in range(10):
            eps=[e for e in plan['episodes'] if e['paired_block']==block]
            for role in [f'V{i}' for i in range(4)]+[f'E{i}' for i in range(4)]:
                present=[e for e in eps if role in e['prompts']]
                self.assertTrue(all(e['prompts'][role]==present[0]['prompts'][role] for e in present))
                self.assertEqual(len({e['seeds'][role] for e in present}),1)
            orders.add(tuple(e['condition'] for e in eps if e['budget']==512))
        self.assertGreater(len(orders),1)
        self.assertEqual(len({e['seeds']['V0'] for e in plan['episodes']}),10)

    def test_changed_workload_or_seed_rejected(self):
        plan=prepare(study_id='offline')
        for kind in ('prompt','seed'):
            changed=copy.deepcopy(plan)
            if kind=='prompt': changed['episodes'][0]['prompts']['V0'].pop()
            else: changed['episodes'][0]['seeds']['V0']+=1
            with self.assertRaisesRegex(ValueError,'fixed design'): validate(changed)

    def test_launcher_has_identical_nuisance_settings_and_disabled_cache(self):
        a,b=command(512),command(2048)
        diff=[(x,y) for x,y in zip(a,b) if x!=y]
        self.assertEqual(diff,[('512','2048')])
        self.assertIn('--no-enable-prefix-caching',a)
        self.assertEqual(a[a.index('--num-gpu-blocks-override')+1],'1024')
        self.assertNotIn('--enable-prefix-caching',a)

    def test_launcher_dry_run_does_not_require_runtime_or_create_log(self):
        with tempfile.TemporaryDirectory() as t:
            log=Path(t)/'server.log'
            p=subprocess.run([sys.executable,str(HERE/'launch_server.py'),'--budget','2048',
                              '--log',str(log),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads(p.stdout)['server_started'])
            self.assertFalse(log.exists())

    def test_main_block_requires_explicit_opt_in_before_runtime_access(self):
        with tempfile.TemporaryDirectory() as t:
            plan=Path(t)/'plan.json'
            plan.write_text(json.dumps(prepare('main',study_id='main',trigger_ms=800,interval_ms=100)))
            out=Path(t)/'output'
            p=subprocess.run([sys.executable,str(HERE/'run.py'),'run-block','--plan',str(plan),
                '--budget','2048','--block','0','--server-log',str(Path(t)/'absent.log'),
                '--output-dir',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('--allow-main-block',p.stderr)
            self.assertFalse(out.exists())


class TimerTests(unittest.TestCase):
    def test_early_wakeup_never_fires_timer_before_absolute_deadline(self):
        current=[0];calls=[]
        async def sleep(seconds):
            calls.append(seconds)
            current[0]+=max(1,int(seconds*1e9/2))
        asyncio.run(sleep_until(800_000_000,now=lambda:current[0],sleep=sleep))
        self.assertGreaterEqual(current[0],800_000_000)
        self.assertGreater(len(calls),1)

    def test_only_explicit_loopback_urls_allowed(self):
        for url in ('https://127.0.0.1:8018','http://example.com:8018','http://127.0.0.1:8018/?q=x'):
            with self.assertRaises(ValueError): local_url(url)


class PromptEchoTests(unittest.TestCase):
    def record(self,payloads):
        return dict(status='complete',raw_sse_events=[dict(raw_data=json.dumps(p),receive_perf_counter_ns=10+i)
            for i,p in enumerate(payloads)]+[dict(raw_data='[DONE]',receive_perf_counter_ns=100)],
            output_token_ids=[103],done_received=True,finish_reason='length',send_ns=5,scheduled_dispatch_ns=4)

    def event(self):
        return dict(choices=[dict(token_ids=[103])],usage=dict(prompt_tokens=2,completion_tokens=1))

    def test_top_level_nested_and_matching_dual_echo_are_accepted(self):
        for position in ('top','nested','both'):
            event=self.event()
            if position!='nested': event['prompt_token_ids']=[101,102]
            if position!='top': event['choices'][0]['prompt_token_ids']=[101,102]
            with self.subTest(position=position):
                self.assertEqual(token_stream(self.record([event]),[101,102],1)['ids'],[103])

    def test_conflicting_dual_or_later_echo_is_rejected(self):
        event=self.event();event['prompt_token_ids']=[101,102]
        event['choices'][0]['prompt_token_ids']=[101,104]
        with self.assertRaisesRegex(ValueError,'contradictory'): token_stream(self.record([event]),[101,102],1)
        event=self.event();event['prompt_token_ids']=[101,102]
        with self.assertRaisesRegex(ValueError,'contradictory'):
            token_stream(self.record([event,dict(choices=[dict(prompt_token_ids=[101,104])])]),[101,102],1)

    def test_invalid_echo_at_either_position_cannot_be_hidden_by_valid_copy(self):
        for invalid in ('101,102',{},[101,True],[101,102.0],[101,-1],[],[101,103]):
            for position in ('top','nested'):
                event=self.event();event['prompt_token_ids']=[101,102]
                event['choices'][0]['prompt_token_ids']=[101,102]
                if position=='top': event['prompt_token_ids']=invalid
                else: event['choices'][0]['prompt_token_ids']=invalid
                with self.subTest(position=position,invalid=invalid),self.assertRaises(ValueError):
                    token_stream(self.record([event]),[101,102],1)

    def test_missing_or_wrong_prompt_echo_is_rejected(self):
        for echo in (None,[101,104]):
            event=self.event();event['choices'][0]['prompt_token_ids']=echo
            with self.assertRaises(ValueError): token_stream(self.record([event]),[101,102],1)


class ConfigurationTests(unittest.TestCase):
    def sample(self,budget=512):
        return dict(http_status=200,error=None,scrape_start_ns=1,response_ns=2,body=server_info(budget))

    def test_resolved_configurations_accept_both_budgets(self):
        for budget in (512,2048): resolved_configuration(self.sample(budget),budget)

    def test_every_checked_runtime_constant_rejects_deviation_and_missing_value(self):
        original=self.sample()
        # Mutate the independently written fixture, not the validation constants.
        for group,fields in original['body']['vllm_config'].items():
            if not isinstance(fields,dict):
                changed=copy.deepcopy(original);changed['body']['vllm_config'][group]={}
                with self.subTest(group=group),self.assertRaises(ValueError): resolved_configuration(changed,512)
                continue
            for key,value in fields.items():
                if isinstance(value,dict):
                    subkey=next(iter(value))
                    changed=copy.deepcopy(original);changed['body']['vllm_config'][group][key][subkey]=9
                    with self.subTest(group=group,key=key),self.assertRaises(ValueError):
                        resolved_configuration(changed,512)
                    continue
                for missing in (False,True):
                    changed=copy.deepcopy(original)
                    if missing: del changed['body']['vllm_config'][group][key]
                    else: changed['body']['vllm_config'][group][key]='adversarial-deviation'
                    with self.subTest(group=group,key=key,missing=missing),self.assertRaises(ValueError):
                        resolved_configuration(changed,512)

    def test_startup_rejects_default_overrides_in_all_parameter_groups(self):
        import ast,re
        raw=startup_text(512)
        match=re.search(r'non-default args: (.*)',raw)
        original=ast.literal_eval(match[1])
        overrides=dict(cpu_offload_gb=1,gpu_memory_utilization=.8,swap_space=8,kv_cache_dtype='fp8',
            kv_cache_memory_bytes=123,kv_offloading_size=1,tensor_parallel_size=2,pipeline_parallel_size=2,
            data_parallel_size=2,data_parallel_size_local=2,decode_context_parallel_size=2,
            prefill_context_parallel_size=2,distributed_executor_backend='ray',enable_expert_parallel=True,
            enable_dbo=True,max_num_partial_prefills=2,max_long_partial_prefills=2,
            long_prefill_token_threshold=128,scheduling_policy='priority',scheduler_cls='custom.Scheduler',
            async_scheduling=False,disable_hybrid_kv_cache_manager=True,stream_interval=2,
            enforce_eager=True,quantization='awq',speculative_config={'model':'other'},
            offload_group_size=1,compilation_config={'mode':0},attention_config={'backend':'TRITON_ATTN'})
        for key,value in overrides.items():
            args=dict(original,**{key:value})
            changed=raw[:match.start(1)]+repr(args)+raw[match.end(1):]
            with self.subTest(key=key),self.assertRaises(ValueError): startup(changed,512)
        startup(raw,512)  # Legitimate omitted 0.17.1 defaults remain accepted.

    def test_startup_engine_and_command_disagreement_are_rejected(self):
        raw=startup_text(512)
        for old,new in [('pipeline_parallel_size=1','pipeline_parallel_size=2'),
                        ('dtype=torch.bfloat16','dtype=torch.float16'),('device_config=cuda','device_config=cpu'),
                        ('kv_cache_dtype=auto','kv_cache_dtype=fp8'),('enforce_eager=False','enforce_eager=True'),
                        ('Using FLASH_ATTN attention','Using TRITON_ATTN attention'),
                        ('Asynchronous scheduling is enabled.','Asynchronous scheduling is disabled.'),
                        ('"--cpu-offload-gb", "0"','"--cpu-offload-gb", "1"')]:
            with self.subTest(old=old),self.assertRaises(ValueError): startup(raw.replace(old,new),512)

    def test_live_cache_labels_reject_changed_and_missing_constants(self):
        import re
        original=metrics(Counter(),0)
        labels=re.search(r'cache_config_info\{(.*)\}',original)[1].split(',')
        for label in labels:
            name=label.split('=')[0]
            for changed in (original.replace(label,name+'="wrong"'),original.replace(label,'')):
                with self.subTest(label=name),self.assertRaises(ValueError):
                    state(dict(http_status=200,error=None,raw_body=changed),512)

    def test_gpu_inventory_and_live_environment_are_strict(self):
        inventory='0, GPU-fixture, NVIDIA GeForce RTX 3090, 24576 MiB, 580.178.04'
        env=dict(CUDA_VISIBLE_DEVICES='0',VLLM_SERVER_DEV_MODE='1')
        gpu_configuration(inventory,env)
        for bad in (inventory.replace('3090','4090'),inventory.replace('24576','12288'),
                    inventory.replace('0, GPU','1, GPU'),inventory+'\n'+inventory):
            with self.assertRaises(ValueError): gpu_configuration(bad,env)
        for key in env:
            with self.assertRaises(ValueError): gpu_configuration(inventory,dict(env,**{key:'wrong'}))

    def test_runtime_binds_cuda_environment_to_logged_api_pid_without_exposing_secrets(self):
        inventory=SimpleNamespace(returncode=0,stdout='0, GPU-fixture, NVIDIA GeForce RTX 3090, 24576 MiB, 580.178.04')
        visited=[]
        environment=['CUDA_VISIBLE_DEVICES=0\0VLLM_SERVER_DEV_MODE=1\0VLLM_API_KEY=do-not-record\0']
        def read(path,*args,**kwargs):
            visited.append(str(path))
            if str(path)=='/proc/4321/environ': return environment[0]
            if str(path)=='/proc/sys/kernel/random/boot_id': return 'offline-boot'
            if str(path)=='/fake-server.log': return startup_text(512)
            raise AssertionError('unexpected read '+str(path))
        with patch('run.importlib.metadata.version',return_value='0.17.1'), \
             patch('run.subprocess.run',return_value=inventory), \
             patch('run.process_start',return_value=1000.25),patch.object(Path,'read_text',read):
            evidence=runtime_evidence(Path('/fake-server.log'),512)
            self.assertIn('/proc/4321/environ',visited)
            self.assertNotIn('do-not-record',json.dumps(evidence))
            self.assertEqual(evidence['server_environment']['CUDA_VISIBLE_DEVICES'],'0')
            environment[0]='CUDA_VISIBLE_DEVICES=1\0VLLM_SERVER_DEV_MODE=1\0'
            with self.assertRaises(ValueError): runtime_evidence(Path('/fake-server.log'),512)


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=prepare(study_id='integration')
        cls.cases={}
        for e in cls.plan['episodes']:
            cls.cases[e['condition']]=execute(e)

    @classmethod
    def tearDownClass(cls):
        for temp,*_ in cls.cases.values(): temp.cleanup()

    def test_all_conditions_capture_full_cohort_and_reconstruct_primary(self):
        for condition,(_,path,result,rows,transport) in self.cases.items():
            with self.subTest(condition=condition):
                self.assertTrue(result['measurement_valid'],result['issues'])
                self.assertTrue(result['all_victims_decoding_at_trigger'])
                self.assertFalse(result['scientific_result_claimed'])
                self.assertEqual(result,read_episode(path))
                self.assertEqual(len(transport.sent),4 if condition=='control' else 8)
                final=rows[-1]['episode']
                stamps=[]
                for role in ('V0','V1','V2','V3'):
                    stamps.append(final['requests'][role]['last_token_receive_ns'])
                self.assertEqual(result['primary_ms'],(max(stamps)-final['victim_start_ns'])/1e6)
                self.assertEqual(final['metrics_before'],final['metrics_samples'][0])
                self.assertEqual(final['metrics_after'],final['metrics_samples'][-1])
                self.assertGreater(len(final['metrics_samples']),2)
                self.assertNotIn('offline-secret',path.read_text())

    def test_declared_burst_and_distributed_deadlines_are_exact(self):
        for condition in ('burst','distributed'):
            final=self.cases[condition][3][-1]['episode']
            extra=[final['requests'][f'E{i}'] for i in range(4)]
            d=[r['scheduled_dispatch_ns'] for r in extra]
            self.assertEqual([b-a for a,b in zip(d,d[1:])],
                             [0]*3 if condition=='burst' else [100_000_000]*3)
            self.assertTrue(all(r['send_ns']>=r['scheduled_dispatch_ns'] for r in extra))
        self.assertGreater(self.cases['distributed'][2]['extra_dispatch_span_ms'],250)
        self.assertLess(self.cases['burst'][2]['extra_dispatch_span_ms'],50)

    def test_independent_copy_needs_no_campaign_file(self):
        with tempfile.TemporaryDirectory() as t:
            copy_path=Path(t)/'alone.jsonl'
            copy_path.write_bytes(self.cases['control'][1].read_bytes())
            self.assertEqual(read_episode(copy_path),self.cases['control'][2])

    def test_raw_stream_timer_and_metric_tampering_are_rejected(self):
        original=self.cases['burst'][3]
        for fault in ('stream','timer','metric','log','role'):
            rows=copy.deepcopy(original)
            if fault=='stream': next(r for r in rows if r['record_type']=='sse_wire')['raw_data']='{}'
            elif fault=='timer': next(r for r in rows if r['record_type']=='timer_trigger')['trigger']['fired_ns']+=1
            elif fault=='metric': next(r for r in rows if r['record_type']=='metrics_sample')['sample']['raw_body']='bad'
            elif fault=='log': next(r for r in rows if r['record_type']=='iteration_log')['raw_text']='bad'
            else: next(r for r in rows if r['record_type']=='request_wire')['role']='foreign'
            with tempfile.TemporaryDirectory() as t:
                path=Path(t)/'tampered.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises((ValueError,KeyError)): read_episode(path)

    def test_partial_episode_is_reported_alongside_valid_episode(self):
        with tempfile.TemporaryDirectory() as t:
            partial=Path(t)/'partial.jsonl'
            partial.write_text(json.dumps(self.cases['control'][3][0])+'\n')
            p=subprocess.run([sys.executable,str(HERE/'run.py'),'audit',str(self.cases['control'][1]),str(partial)],
                             capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            report=json.loads(p.stdout)
            self.assertEqual(len(report),2)
            self.assertEqual(report[1]['status'],'incomplete_or_invalid')

    def test_output_never_overwrites_existing_episode(self):
        with self.assertRaises(FileExistsError): Journal(self.cases['control'][1])

    def test_counter_reset_is_not_interpreted_as_zero_load(self):
        import re
        episode=copy.deepcopy(self.cases['burst'][3][-1]['episode'])
        episode['metrics_samples'][-1]['raw_body']=re.sub(
            r'vllm:prompt_tokens_total \d+', 'vllm:prompt_tokens_total 0',
            episode['metrics_samples'][-1]['raw_body'])
        episode['metrics_after']=episode['metrics_samples'][-1]
        report=assess(episode['plan'],episode)
        self.assertFalse(report['measurement_valid'])
        self.assertTrue(any('counter reset' in reason for reason in report['issues']))

    def test_iteration_over_budget_is_rejected_even_with_balanced_context_sum(self):
        episode=copy.deepcopy(self.cases['burst'][3][-1]['episode'])
        raw=episode['iteration_log'].replace('2044 context tokens','2050 context tokens',1)
        raw=raw.replace('256 context tokens','250 context tokens',1)
        self.assertNotEqual(raw,episode['iteration_log'])
        episode['iteration_log']=raw
        episode['iteration_log_sha256']=hashlib.sha256(raw.encode()).hexdigest()
        report=assess(episode['plan'],episode)
        self.assertFalse(report['measurement_valid'])
        self.assertIn('scheduler budget exceeded',report['issues'])

    def test_configuration_snapshot_tampering_fails_independent_integrity_audit(self):
        rows=copy.deepcopy(self.cases['burst'][3])
        next(r for r in rows if r['record_type']=='server_config')['sample']['body']['vllm_config']['parallel_config']['tensor_parallel_size']=2
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'tampered.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'configuration journal/copy mismatch'): read_episode(path)

    def test_resolved_configuration_drift_outside_named_subset_is_also_rejected(self):
        episode=copy.deepcopy(self.cases['burst'][3][-1]['episode'])
        episode['server_config_after']['body']['vllm_config']['instance_id']='changed-instance'
        result=assess(episode['plan'],episode)
        self.assertFalse(result['measurement_valid'])
        self.assertTrue(any('changed during episode' in x for x in result['issues']))


class FailureTests(unittest.TestCase):
    def run_fault(self,fault,condition='burst'):
        plan=next(e for e in prepare(study_id='failure')['episodes'] if e['condition']==condition)
        case=execute(plan,fault)
        self.addCleanup(case[0].cleanup)
        return case

    def test_early_victim_completion_does_not_prevent_timer_or_extra_requests(self):
        _,path,result,rows,transport=self.run_fault('early','distributed')
        self.assertTrue(result['measurement_valid'],result['issues'])
        self.assertFalse(result['all_victims_decoding_at_trigger'])
        self.assertFalse(result['eligible_for_planned_comparison'])
        self.assertEqual(len(transport.sent),8)
        e=rows[-1]['episode']
        self.assertGreaterEqual(e['trigger']['fired_ns'],e['victim_start_ns']+800_000_000)
        self.assertTrue(all(stamp>=e['trigger']['deadline_ns'] for role,stamp,_ in transport.sent if role.startswith('E')))

    def test_nested_prompt_echo_survives_real_executor_and_independent_audit(self):
        _,path,result,_,_=self.run_fault('nested_echo')
        self.assertTrue(result['measurement_valid'],result['issues'])
        self.assertEqual(result,read_episode(path))

    def test_diagnostic_512_all_conditions_pass_same_full_episode_audit(self):
        for plan in prepare(study_id='diagnostic512',diagnostic_budget=512)['episodes']:
            case=execute(plan)
            self.addCleanup(case[0].cleanup)
            with self.subTest(condition=plan['condition']):
                self.assertTrue(case[2]['eligible_for_planned_comparison'],case[2]['issues'])
                self.assertEqual(case[2],read_episode(case[1]))
                self.assertEqual(case[2]['mode'],'diagnostic')
                self.assertLessEqual(case[2]['scheduler']['max_scheduled_tokens'],512)

    def test_invalid_or_unavailable_resolved_config_stops_before_victims(self):
        for fault in ('config_before','config_http'):
            _,path,result,rows,transport=self.run_fault(fault)
            self.assertEqual(transport.sent,[])
            self.assertFalse(result['measurement_valid'])
            self.assertFalse(read_episode(path)['measurement_valid'])
            self.assertTrue(any('configuration' in x for x in result['issues']))

    def test_after_configuration_deviation_invalidates_retained_episode(self):
        _,path,result,_,transport=self.run_fault('config_after')
        self.assertEqual(len(transport.sent),8)
        self.assertFalse(result['measurement_valid'])
        self.assertFalse(read_episode(path)['measurement_valid'])

    def test_missing_done_keeps_episode_but_invalidates_cohort_endpoint(self):
        _,path,result,rows,_=self.run_fault('missing_done')
        self.assertFalse(result['measurement_valid'])
        self.assertIsNone(result['primary_ms'])
        self.assertEqual(rows[-1]['episode']['requests']['V1']['status'],'incomplete')
        self.assertEqual(read_episode(path),result)

    def test_timeout_keeps_partial_sse_and_error_status(self):
        _,path,result,rows,_=self.run_fault('timeout')
        victim=rows[-1]['episode']['requests']['V1']
        self.assertTrue(victim['timed_out'])
        self.assertTrue(victim['raw_sse_events'])
        self.assertIsNone(result['primary_ms'])
        self.assertFalse(read_episode(path)['measurement_valid'])

    def test_metric_gap_and_failed_scrape_invalidate_coverage(self):
        for fault in ('gap','metrics_http'):
            _,path,result,rows,_=self.run_fault(fault)
            self.assertFalse(result['measurement_valid'])
            self.assertTrue(any('metric' in issue for issue in result['issues']),result['issues'])
            self.assertGreater(len(rows[-1]['episode']['metrics_samples']),2)
            self.assertFalse(read_episode(path)['measurement_valid'])

    def test_effective_budget_mismatch_is_rejected(self):
        with self.assertRaises(ValueError): startup(startup_text(2048),512)

    def test_budget_512_uses_same_workload_and_respects_actual_iteration_budget(self):
        plan=next(e for e in prepare('main',study_id='b512',trigger_ms=800,interval_ms=100)['episodes']
                  if e['budget']==512 and e['paired_block']==0 and e['condition']=='burst')
        case=execute(plan)
        self.addCleanup(case[0].cleanup)
        result=case[2]
        self.assertTrue(result['measurement_valid'],result['issues'])
        self.assertEqual(result['scheduler']['max_scheduled_tokens'],512)
        self.assertTrue(result['scheduler']['budget_binding_observed'])

    def test_wrong_server_stops_before_any_victim_request(self):
        _,path,result,rows,transport=self.run_fault('wrong_server')
        self.assertEqual(transport.sent,[])
        self.assertFalse(result['measurement_valid'])
        self.assertIsNone(result['primary_ms'])
        self.assertTrue(rows[-1]['episode']['errors'])
        self.assertFalse(read_episode(path)['measurement_valid'])

    def test_log_rotation_is_retained_as_an_invalid_episode(self):
        _,path,result,rows,_=self.run_fault('rotation')
        self.assertFalse(result['measurement_valid'])
        self.assertTrue(any(e['detail']=='iteration log unavailable' for e in rows[-1]['episode']['errors']))
        self.assertFalse(read_episode(path)['measurement_valid'])

    def test_cancelled_episode_is_saved_and_cancellation_propagates(self):
        plan=prepare(study_id='cancel')['episodes'][0]
        with tempfile.TemporaryDirectory() as t:
            path=Path(t);log=path/'server.log';raw=startup_text(2048);log.write_text(raw)
            runtime=dict(startup_log=raw,startup_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                         api_pid=4321,process_start=1000.25)
            journal=Journal(path/'cancel.jsonl');transport=FakeTransport(plan,log)
            async def go():
                task=asyncio.create_task(run_episode(load_base(),transport,plan,journal,runtime,log,
                    base_url='http://127.0.0.1:8018',api_key='offline-secret',origin='synthetic'))
                await asyncio.sleep(.05)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
            try: asyncio.run(go())
            finally: journal.close()
            rows=[json.loads(s) for s in (path/'cancel.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['record_type'],'episode_result')
            self.assertFalse(rows[-1]['assessment']['measurement_valid'])
            self.assertTrue(all(role.startswith('V') for role,_,_ in transport.sent))

    def test_block_preflight_failure_lists_every_not_started_episode(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'block'
            with patch('run.runtime_evidence',side_effect=ValueError('preflight failed')):
                report=asyncio.run(run_block(prepare(study_id='failed-block'),2048,0,
                    Path(t)/'absent.log',out,'http://127.0.0.1:8018','offline-secret'))
            self.assertEqual(len(report['not_completed']),3)
            self.assertTrue((out/'block_summary.json').exists())


class TransportTests(unittest.TestCase):
    def test_shared_pool_keeps_per_request_timeouts_and_closes_once(self):
        calls=[]
        class Client:
            def __init__(self,**kwargs): self.options=kwargs
            async def get(self,*args,**kwargs): calls.append(('get',kwargs['timeout'])); return object()
            def stream(self,*args,**kwargs): calls.append(('stream',kwargs['timeout'])); return object()
            async def aclose(self): calls.append(('close',None))
        module=SimpleNamespace(AsyncClient=Client,Limits=lambda **kwargs:kwargs)
        base=SimpleNamespace(HttpxTransport=lambda:SimpleNamespace(_httpx=module))
        async def go():
            transport=await shared_httpx_transport(base)
            async with transport._httpx.AsyncClient(timeout=2) as c: await c.get('local')
            async with transport._httpx.AsyncClient(timeout=60) as c: c.stream('POST','local')
            await transport.close()
        asyncio.run(go())
        self.assertEqual(calls,[('get',2),('stream',60),('close',None)])

    def test_disk_failure_prevents_successful_flush(self):
        with tempfile.TemporaryDirectory() as t:
            journal=Journal(Path(t)/'raw.jsonl')
            with patch('instrument.os.fsync',side_effect=OSError('disk failure')):
                journal.emit('test')
                with self.assertRaisesRegex(ValueError,'persistence'): asyncio.run(journal.flush())
            with self.assertRaisesRegex(ValueError,'persistence'): journal.close()


if __name__=='__main__':
    unittest.main()
