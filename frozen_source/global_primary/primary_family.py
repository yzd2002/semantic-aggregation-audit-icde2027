"""Explicit prospective nine-case primary analysis; no model or data I/O."""
import hashlib
import numpy as np
BASE_SEED=2026110613
TASKS=('dblp_acm','boolq','snli')
MODELS=('qwen','smol','glm')
PRIMARY_CASES=tuple((task,model) for task in TASKS for model in MODELS)
BOOTSTRAP_DRAWS=4000
LOWER_QUANTILE=1/360
UPPER_QUANTILE=359/360
NOMINAL_CONFIDENCE=179/180

def case_seed(task,model):
 if (task,model) not in PRIMARY_CASES:raise ValueError('Unknown prospective primary case')
 return int.from_bytes(hashlib.sha256(f'{BASE_SEED}|{task}|{model}'.encode('utf-8')).digest()[:8],'big')

def excess_vector(near1,near2,random1,random2):
 arrays=[np.asarray(v,dtype=float) for v in (near1,near2,random1,random2)]
 if any(v.ndim!=1 or v.shape!=arrays[0].shape or not np.all(np.isin(v,[0,1])) for v in arrays):raise ValueError('Supply aligned complete binary targets only')
 a1,a2,b1,b2=arrays
 return ((a1!=b1).astype(float)+(a2!=b2).astype(float)-(a1!=a2).astype(float)-(b1!=b2).astype(float))/2

def primary_interval(values,*,task,model,planned_n=128,mode='primary_mixed'):
 seed=case_seed(task,model)
 if mode!='primary_mixed' or planned_n!=128:raise ValueError('This primary family uses mixed execution and fixed 128-target relations')
 x=np.asarray(values,dtype=float)
 if x.ndim!=1 or len(x)>planned_n or not np.all(np.isfinite(x)) or not np.all(np.isin(x,[-1,0,1])):raise ValueError('Invalid common12 excess vector')
 result={'task':task,'model':model,'mode':mode,'scope':'full-first','planned_target_n':planned_n,
  'common12_complete_n':len(x),'excluded_n':planned_n-len(x),'bootstrap_draws':BOOTSTRAP_DRAWS,
  'bootstrap_seed':seed,'confidence':NOMINAL_CONFIDENCE,'quantiles':[LOWER_QUANTILE,UPPER_QUANTILE],
  'nominal_family_alpha':.05,'primary_family_size':9,'exact_population_coverage_claimed':False,
  'exact_FWER_claimed':False,'legacy_ci98_75_authoritative_for_this_family':False}
 if len(x)==0:return {**result,'available':False,'mean_excess':None,'interval':None,'positive_primary_evidence':False}
 indices=np.random.default_rng(seed).integers(0,len(x),size=(BOOTSTRAP_DRAWS,len(x)))
 draws=x[indices].mean(axis=1);lo,hi=np.quantile(draws,[LOWER_QUANTILE,UPPER_QUANTILE],method='linear')
 return {**result,'available':True,'mean_excess':float(x.mean()),'interval':[float(lo),float(hi)],'positive_primary_evidence':bool(lo>0)}

def assemble_suite(cases):
 values={(r['task'],r['model']):r for r in cases}
 if len(cases)!=9 or len(values)!=9 or set(values)!=set(PRIMARY_CASES):raise ValueError('Exactly all nine primary cases are required, including unavailable cases')
 for r in cases:
  if r['confidence']!=NOMINAL_CONFIDENCE or r['primary_family_size']!=9 or r['mode']!='primary_mixed':raise ValueError('Primary-family settings differ')
 return {'family':'fresh three-task three-model full-first mixed suite','nominal_family_alpha':.05,
  'per_case_confidence':NOMINAL_CONFIDENCE,'all_nine_cases_reported':True,
  'approximate_selected_target_bootstrap':True,'cases':[values[k] for k in PRIMARY_CASES]}
