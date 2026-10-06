"""Read-only reanalysis of the two approved existing-data questions.

Consumes compact numeric extracts from immutable Git blobs; does not issue
LLM requests, start a server, or implement workload generation.
"""
import csv, json, math, statistics as st
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)

def med(xs): return st.median(xs)
def gm(xs): return math.exp(st.mean(math.log(x) for x in xs))
def ms(ns): return ns/1e6
def s(ns): return ns/1e9

def stream(r):
    return [e for e in r['events'] if e['tokens']>0]

def max_gap(r):
    ev=stream(r)
    return max((ms(b['ns']-a['ns']) for a,b in zip(ev,ev[1:])),default=0.0)

def request_tpot(r):
    return ms(r['last_token_receive_ns']-r['first_token_receive_ns'])/(r['output_token_count']-1)

def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def budget_metrics(e):
    active=[r for r in e['victims'] if r['request_index']<4]
    return {
        'active_e2el_ms':med([r['e2el_ms'] for r in active]),
        'active_tpot_ms':med([request_tpot(r) for r in active]),
        'active_max_gap_ms':med([max_gap(r) for r in active]),
        'active_worst_gap_ms':max(max_gap(r) for r in active),
        'victim_completion_span_ms':ms(max(r['stream_end_ns'] for r in e['victims'])-min(r['request_start_ns'] for r in e['victims'])),
        'victim_output_span_ms':ms(max(r['last_token_receive_ns'] for r in e['victims'])-min(r['request_start_ns'] for r in e['victims'])),
    }

def validation(episodes,pairs):
    errors=[]; batch=Counter(); baseline_pairs=[]; tpot_discrepancies=[]
    for e in episodes:
        if e['status']!='complete' or e['validation_errors'] or e['run_mode']!='official':errors.append([e['episode_id'],'episode status'])
        if len(e['victims'])!=20:errors.append([e['episode_id'],'victim count'])
        for r in e['victims']+e['bursts']:
            ev=stream(r)
            if r['status']!='complete' or r['validation_errors']:errors.append([e['episode_id'],r['request_index'],'request status'])
            if sum(v['tokens'] for v in ev)!=r['output_token_count']:errors.append([e['episode_id'],r['request_index'],'SSE count mismatch'])
            if r['output_token_count']!=r['expected_completion_tokens']:errors.append([e['episode_id'],r['request_index'],'output count'])
            if not ev or ev[0]['ns']!=r['first_token_receive_ns'] or ev[-1]['ns']!=r['last_token_receive_ns']:errors.append([e['episode_id'],r['request_index'],'token timestamps'])
            if not r['request_start_ns']<=r['first_token_receive_ns']<=r['last_token_receive_ns']<=r['stream_end_ns']:errors.append([e['episode_id'],r['request_index'],'timestamp ordering'])
            if any(a['ns']>b['ns'] for a,b in zip(ev,ev[1:])):errors.append([e['episode_id'],r['request_index'],'nonmonotone events'])
            batch.update(v['tokens'] for v in ev)
            if r.get('client_observed_tpot_ms') is not None:
                diff=abs(request_tpot(r)-r['client_observed_tpot_ms'])
                if diff>1e-6:tpot_discrepancies.append([e['episode_id'],r['request_index'],diff])
    for block,p in pairs.items():
        if set(p)!= {'no_burst','prefill_burst'}:errors.append([block,'incomplete pair']);continue
        c,b=p['no_burst'],p['prefill_burst']
        cs,bs=c['schedule'],b['schedule']
        for key in ['model','model_key','offload_gb','concurrency','server_max_num_seqs','max_num_batched_tokens','repeat','victim_workload_seed','burst_workload_seed','victim_input_len','victim_output_len','burst_input_len','burst_output_len']:
            if cs.get(key)!=bs.get(key):errors.append([block,key,'schedule mismatch'])
        cr={r['request_index']:r for r in c['victims']}; br={r['request_index']:r for r in b['victims']}
        for i in cr:
            for key in ['prompt_sha256','prompt_seed','generation_seed']:
                if cr[i].get(key)!=br[i].get(key):errors.append([block,i,key,'request pairing mismatch'])
        baseline_pairs.append({'block':block,'same_prompt_hashes':all(cr[i].get('prompt_sha256')==br[i].get('prompt_sha256') for i in cr),'control_first':cs.get('within_block_order',cs.get('order_in_block'))==1})
    return {'episode_count':len(episodes),'pair_count':len(pairs),'errors':errors,'all_pairs_same_prompts':all(p['same_prompt_hashes'] for p in baseline_pairs),'control_first_pairs':sum(p['control_first'] for p in baseline_pairs),'token_batch_size_counts':dict(batch),'tpot_reconstruction_discrepancies':tpot_discrepancies}

def class_of(e,endpoint):
    first=min(r['first_token_receive_ns'] for r in e['bursts'])
    ix=set(e['trigger']['cohort_indices'])
    active=max(r[endpoint] for r in e['victims'] if r['request_index'] in ix)
    final=max(r[endpoint] for r in e['victims'])
    return 'A' if first<=active else ('B' if first<=final else 'C')

def queue_row(e,c):
    vs=e['victims'];k=e['schedule']['server_max_num_seqs'];n=len(vs);m=len(e['bursts'])
    first=min(r['first_token_receive_ns'] for r in e['bursts'])
    first_dispatch=min(r['request_start_ns'] for r in e['bursts'])
    row={'episode_id':e['episode_id'],'model':e['schedule']['model_key'],'offload_gb':e['schedule']['offload_gb'],'k':k,'repeat':e['schedule']['repeat'],'n':n,'burst_count':m,'simple_wave_prediction':'C' if n%k==0 else 'B','class_stream_end':class_of(e,'stream_end_ns'),'class_last_token':class_of(e,'last_token_receive_ns'),'metrics_quality':e['trigger']['metrics_quality'],'metrics_running':e['trigger']['metrics_running'],'metrics_waiting':e['trigger']['metrics_waiting'],'all_victims_dispatched_before_burst':max(r['request_start_ns'] for r in vs)<first_dispatch,'burst_dispatched_before_first_victim_completion':first_dispatch<min(r['stream_end_ns'] for r in vs),'metrics_end_before_burst_dispatch':e['trigger']['metrics_last_ns']<first_dispatch}
    for endpoint,label in [('stream_end_ns','end'),('last_token_receive_ns','token')]:
        ordered=sorted(vs,key=lambda r:r[endpoint]); ordered_c=sorted(c['victims'],key=lambda r:r[endpoint])
        rank=n-k+1
        release=ordered[rank-1][endpoint]
        last_admission_rank=min(n,n-k+m)
        last_release=ordered[last_admission_rank-1][endpoint]
        final=ordered[-1][endpoint]
        initial_active=set(e['trigger']['cohort_indices'])
        active_end=max(r[endpoint] for r in vs if r['request_index'] in initial_active)
        row.update({f'fcfs_first_release_rank_{label}':rank,f'first_output_after_first_release_s_{label}':s(first-release),f'first_output_after_last_burst_release_s_{label}':s(first-last_release),f'first_output_after_all_victims_s_{label}':s(first-final),f'first_output_after_initial_cohort_s_{label}':s(first-active_end),f'remaining_victim_time_at_first_release_s_{label}':s(final-release),f'last_k_completion_spread_s_{label}':s(ordered[-1][endpoint]-ordered[-k][endpoint]),f'final_k_request_indices_{label}':','.join(str(r['request_index']) for r in ordered[-k:]),f'wave_boundary_spreads_s_{label}':';'.join(str(s(ordered[j-1][endpoint]-ordered[max(0,j-k)][endpoint])) for j in range(k,n+1,k)),f'control_last_k_completion_spread_s_{label}':s(ordered_c[-1][endpoint]-ordered_c[-k][endpoint])})
    row['observable_complete_victims_before_first_burst_output']=sum(r['stream_end_ns']<first for r in vs)
    row['observable_complete_victims_before_first_burst_dispatch']=sum(r['stream_end_ns']<first_dispatch for r in vs)
    return row

def main():
    data=json.loads((ROOT/'input_compact.json').read_text());episodes=data['episodes']
    pairs=defaultdict(dict)
    for e in episodes:
        assert e['schedule']['condition'] not in pairs[e['block_id']], 'duplicate episode'
        pairs[e['block_id']][e['schedule']['condition']]=e
    valid=validation(episodes,pairs)
    (OUT/'validation.json').write_text(json.dumps(valid,indent=2))
    if valid['errors'] or valid['tpot_reconstruction_discrepancies']:raise RuntimeError(valid)
    budget_pairs=[];queue=[]
    for block,p in sorted(pairs.items()):
        c,b=p['no_burst'],p['prefill_burst']
        if '/chunk_budget_screen/' in b['path']:
            cm,bm=budget_metrics(c),budget_metrics(b)
            row={'block_id':block,'offload_gb':b['schedule']['offload_gb'],'budget':b['schedule']['max_num_batched_tokens'],'repeat':b['schedule']['repeat'],'control_first':c['schedule']['within_block_order']==1}
            for key in cm:
                row['control_'+key]=cm[key];row['loaded_'+key]=bm[key];row['delta_'+key]=bm[key]-cm[key];row['ratio_'+key]=bm[key]/cm[key]
            active_b=[r for r in b['victims'] if r['request_index']<4]
            row['active_tokens_at_burst_dispatch_min']=min(sum(v['tokens'] for v in stream(r) if v['ns']<=min(q['request_start_ns'] for q in b['bursts'])) for r in active_b)
            row['active_tokens_at_burst_dispatch_max']=max(sum(v['tokens'] for v in stream(r) if v['ns']<=min(q['request_start_ns'] for q in b['bursts'])) for r in active_b)
            row['burst_completion_span_ms']=ms(max(r['stream_end_ns'] for r in b['bursts'])-min(r['request_start_ns'] for r in b['bursts']))
            budget_pairs.append(row)
        else:
            queue.append(queue_row(b,c))
    write_csv('budget_pairs.csv',budget_pairs);write_csv('queue_episodes.csv',queue)
    bg=defaultdict(list)
    for r in budget_pairs:bg[(r['offload_gb'],r['budget'])].append(r)
    budget_summary=[]
    for (off,budget),rs in sorted(bg.items()):
        row={'offload_gb':off,'budget':budget,'n':len(rs)}
        for key in ['active_e2el_ms','active_tpot_ms','active_max_gap_ms','active_worst_gap_ms','victim_completion_span_ms','victim_output_span_ms']:
            row['control_median_'+key]=med([r['control_'+key] for r in rs]);row['loaded_median_'+key]=med([r['loaded_'+key] for r in rs]);row['median_delta_'+key]=med([r['delta_'+key] for r in rs]);row['delta_min_'+key]=min(r['delta_'+key] for r in rs);row['delta_max_'+key]=max(r['delta_'+key] for r in rs);row['geomean_ratio_'+key]=gm([r['ratio_'+key] for r in rs])
        row['burst_completion_span_median_ms']=med([r['burst_completion_span_ms'] for r in rs])
        budget_summary.append(row)
    write_csv('budget_summary.csv',budget_summary)
    # Same repeat uses the same workload seeds across budget settings.
    # Verify actual per-index workload identities before forming contrasts.
    budget_identity_checks=[]
    for off in (0,12):
        for rep in (1,2,3):
            subset=[e for e in episodes if '/chunk_budget_screen/' in e['path'] and e['schedule']['offload_gb']==off and e['schedule']['repeat']==rep]
            ref=subset[0]
            signature=lambda e: sorted((r['request_index'],r.get('prompt_sha256'),r.get('prompt_seed'),r.get('generation_seed')) for r in e['victims'])
            assert len(subset)==6 and all(signature(e)==signature(ref) for e in subset), 'cross-budget workloads differ'
            budget_identity_checks.append({'offload_gb':off,'repeat':rep,'same_victim_workloads_across_budgets':True})
    contrasts=[]
    for off in (0,12):
        for budget in (512,1024):
            for rep in (1,2,3):
                low=next(r for r in budget_pairs if (r['offload_gb'],r['budget'],r['repeat'])==(off,budget,rep))
                base=next(r for r in budget_pairs if (r['offload_gb'],r['budget'],r['repeat'])==(off,2048,rep))
                row={'offload_gb':off,'budget':budget,'repeat':rep}
                for metric in ('active_e2el_ms','active_max_gap_ms','active_tpot_ms','victim_completion_span_ms'):
                    row['paired_effect_contrast_'+metric]=low['delta_'+metric]-base['delta_'+metric]
                row['loaded_max_gap_ratio']=low['loaded_active_max_gap_ms']/base['loaded_active_max_gap_ms']
                row['burst_completion_span_ratio']=low['burst_completion_span_ms']/base['burst_completion_span_ms']
                contrasts.append(row)
    write_csv('budget_contrasts_vs2048.csv',contrasts)
    contrast_summary=[]
    for off in (0,12):
        for budget in (512,1024):
            rs=[r for r in contrasts if r['offload_gb']==off and r['budget']==budget]
            row={'offload_gb':off,'budget':budget,'n':len(rs)}
            for key in rs[0]:
                if key in ('offload_gb','budget','repeat'): continue
                vals=[r[key] for r in rs]
                row[key+'_median']=med(vals);row[key+'_min']=min(vals);row[key+'_max']=max(vals)
            contrast_summary.append(row)
    write_csv('budget_contrast_summary.csv',contrast_summary)
    qg=defaultdict(list)
    for r in queue:qg[(r['model'],r['offload_gb'],r['k'])].append(r)
    queue_summary=[]
    for (model,off,k),rs in sorted(qg.items()):
        row={'model':model,'offload_gb':off,'k':k,'n':len(rs),'class_stream_end':dict(Counter(r['class_stream_end'] for r in rs)),'class_last_token':dict(Counter(r['class_last_token'] for r in rs)),'all_match_wave_prediction':all(r['class_stream_end']==r['simple_wave_prediction'] for r in rs),'first_release_proxy_violations_end':sum(r['first_output_after_first_release_s_end']<0 for r in rs),'first_release_proxy_violations_token':sum(r['first_output_after_first_release_s_token']<0 for r in rs)}
        for key in ['first_output_after_first_release_s_end','first_output_after_last_burst_release_s_end','first_output_after_all_victims_s_end','remaining_victim_time_at_first_release_s_end','last_k_completion_spread_s_end','control_last_k_completion_spread_s_end']:
            row[key+'_median']=med([r[key] for r in rs]);row[key+'_min']=min(r[key] for r in rs);row[key+'_max']=max(r[key] for r in rs)
        queue_summary.append(row)
    write_csv('queue_summary.csv',queue_summary)
    result={'source_ref':data['ref'],'validation':valid,'budget_summary':budget_summary,'budget_identity_checks':budget_identity_checks,'budget_contrast_summary':contrast_summary,'queue_summary':queue_summary,'budget_pair_count':len(budget_pairs),'queue_episode_count':len(queue),'queue_proxy_violations':sum(r['first_output_after_first_release_s_end']<0 for r in queue),'queue_class_endpoint_disagreements':sum(r['class_stream_end']!=r['class_last_token'] for r in queue),'metrics_all_only_pre_burst':all(r['metrics_end_before_burst_dispatch'] for r in queue)}
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'validation':valid,'budget_pairs':len(budget_pairs),'queue_episodes':len(queue),'queue_proxy_violations':result['queue_proxy_violations'],'queue_class_endpoint_disagreements':result['queue_class_endpoint_disagreements'],'metrics_all_only_pre_burst':result['metrics_all_only_pre_burst']},indent=2))

if __name__=='__main__':main()
