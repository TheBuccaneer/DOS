#!/usr/bin/env python3
import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path

def med(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(xs) if xs else None

def pct(xs,q):
    xs=sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs: return None
    if len(xs)==1: return xs[0]
    p=(len(xs)-1)*q; a=int(p); b=min(a+1,len(xs)-1)
    return xs[a]+(xs[b]-xs[a])*(p-a)

def div(a,b): return a/b if a is not None and b not in (None,0) else None

def rnd(x):
    if isinstance(x,float): return round(x,4)
    if isinstance(x,list): return [rnd(v) for v in x]
    if isinstance(x,dict): return {k:rnd(v) for k,v in x.items()}
    return x

def intervals(req):
    times=[]
    for e in req.get('raw_sse_events',[]):
        ids=e.get('token_ids'); t=e.get('receive_perf_counter_ns')
        if ids:
            if len(ids)!=1 or not isinstance(t,int): return []
            times.append(t)
    return [(times[i-1],times[i],(times[i]-times[i-1])/1e6) for i in range(1,len(times))]

def wave(req,conc): return 'active' if req.get('request_index',999999)<conc else 'later'

def window_stats(reqs,conc,which,start,end):
    vals=[]; reqmax=[]; exposed=0
    for r in reqs:
        if r.get('status')!='complete' or wave(r,conc)!=which: continue
        v=[d for a,b,d in intervals(r) if b>start and a<end]
        if v:
            exposed+=1; vals+=v; reqmax.append(max(v))
    return {'exposed_requests':exposed,'intervals':len(vals),'median_itl_ms':med(vals),
            'p95_itl_ms':pct(vals,.95),'max_itl_ms':max(vals) if vals else None,
            'median_request_max_itl_ms':med(reqmax)}

def full_stats(reqs,conc,which):
    rs=[r for r in reqs if r.get('status')=='complete' and wave(r,conc)==which]
    return {'requests':len(rs),'median_tpot_ms':med(r.get('client_observed_tpot_ms') for r in rs),
            'median_ttft_ms':med(r.get('ttft_ms') for r in rs),
            'median_e2el_ms':med(r.get('e2el_ms') for r in rs),
            'median_max_itl_ms':med(max(r['itl_ms']) for r in rs if r.get('itl_ms'))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir',type=Path)
    a=ap.parse_args()
    files=sorted((a.run_dir/'episodes').glob('*.json'))
    if not files: raise SystemExit('Keine JSON-Dateien in RUN_DIR/episodes gefunden')
    eps=[json.loads(p.read_text()) for p in files]
    groups=defaultdict(dict)
    for e in eps:
        s=e['schedule_row']
        key=(s['model_key'],s['state_label'],s['offload_gb'],s['concurrency'],s['trigger_after_decode_tokens'],s['repeat'])
        groups[key][s['condition']]=e
    pairs=[]
    for key,g in sorted(groups.items(),key=str):
        if not {'no_burst','prefill_burst'}<=g.keys(): continue
        c,b=g['no_burst'],g['prefill_burst']; s=b['schedule_row']; conc=s['concurrency']
        bi=b['burst_interval']; dur=bi['end_ns']-bi['start_ns']; ct=c['trigger']['trigger_perf_ns']
        ac=window_stats(c['victim_requests'],conc,'active',ct,ct+dur)
        ab=window_stats(b['victim_requests'],conc,'active',bi['start_ns'],bi['end_ns'])
        lc=window_stats(c['victim_requests'],conc,'later',ct,ct+dur)
        lb=window_stats(b['victim_requests'],conc,'later',bi['start_ns'],bi['end_ns'])
        fc=full_stats(c['victim_requests'],conc,'active'); fb=full_stats(b['victim_requests'],conc,'active')
        pairs.append({'state':s['state_label'],'offload_gb':s['offload_gb'],'trigger_tokens':s['trigger_after_decode_tokens'],
          'repeat':s['repeat'],'burst_duration_ms':dur/1e6,
          'active':{'control':ac,'burst':ab,'median_ratio':div(ab['median_itl_ms'],ac['median_itl_ms']),
                    'p95_ratio':div(ab['p95_itl_ms'],ac['p95_itl_ms']),
                    'max_ratio':div(ab['max_itl_ms'],ac['max_itl_ms'])},
          'later':{'control':lc,'burst':lb,'median_ratio':div(lb['median_itl_ms'],lc['median_itl_ms']),
                   'p95_ratio':div(lb['p95_itl_ms'],lc['p95_itl_ms']),
                   'max_ratio':div(lb['max_itl_ms'],lc['max_itl_ms'])},
          'active_full_tpot_ratio':div(fb['median_tpot_ms'],fc['median_tpot_ms']),
          'trigger_skew_ms':{'control':c['trigger'].get('trigger_crossing_skew_ms'),'burst':b['trigger'].get('trigger_crossing_skew_ms')}})
    agg=[]
    by=defaultdict(list)
    for p in pairs: by[(p['state'],p['trigger_tokens'])].append(p)
    for (state,trig),rs in sorted(by.items(),key=lambda x:(x[0][1],x[0][0])):
        agg.append({'state':state,'trigger_tokens':trig,'pairs':len(rs),
          'active_median_itl_ratio':med(r['active']['median_ratio'] for r in rs),
          'active_p95_itl_ratio':med(r['active']['p95_ratio'] for r in rs),
          'active_max_itl_ratio':med(r['active']['max_ratio'] for r in rs),
          'later_median_itl_ratio':med(r['later']['median_ratio'] for r in rs),
          'later_max_itl_ratio':med(r['later']['max_ratio'] for r in rs),
          'active_full_tpot_ratio':med(r['active_full_tpot_ratio'] for r in rs)})
    mods=[]; bt=defaultdict(dict)
    for r in agg: bt[r['trigger_tokens']][r['state']]=r
    for trig,d in sorted(bt.items()):
        if 'low' in d and 'high' in d:
            mods.append({'trigger_tokens':trig,
              'high_over_low_active_median_itl_ratio':div(d['high']['active_median_itl_ratio'],d['low']['active_median_itl_ratio']),
              'high_over_low_active_p95_itl_ratio':div(d['high']['active_p95_itl_ratio'],d['low']['active_p95_itl_ratio']),
              'high_over_low_active_max_itl_ratio':div(d['high']['active_max_itl_ratio'],d['low']['active_max_itl_ratio']),
              'high_over_low_active_full_tpot_ratio':div(d['high']['active_full_tpot_ratio'],d['low']['active_full_tpot_ratio'])})
    out={'episode_count':len(eps),'complete_count':sum(e.get('status')=='complete' for e in eps),
         'fingerprints':sorted(set(e.get('schedule_fingerprint') for e in eps)),
         'matched_pairs':pairs,'aggregate_by_state_trigger':agg,'state_modulation_high_over_low':mods}
    print(json.dumps(rnd(out),indent=2))
if __name__=='__main__': main()
