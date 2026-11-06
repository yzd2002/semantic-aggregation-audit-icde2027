"""Byte-for-byte extracted metric functions, bound by provenance; no model imports."""
import math
from fractions import Fraction

def complete(values,n=128):
 return len(values)==n and all(type(v) is int and v in (0,1) for v in values)

def disagreement(pred,reference,n=128):
 okay=complete(pred,n) and complete(reference,n)
 errors=sum(a!=b for a,b in zip(pred,reference)) if okay else None
 return {'complete':okay,'numerator':errors,'fixed_denominator':n,'value':errors/n if okay else None}

def gold_metrics(pred,gold,n=128):
 assert complete(gold,n) and len(pred)==n
 counts={name:0 for name in ('tp','tn','fp','fn')}
 for p,g in zip(pred,gold):
  if type(p) is int and p in (0,1):counts[('tn','fp','fn','tp')[2*g+p]]+=1
 okay=complete(pred,n);valid=sum(counts.values());positive=sum(gold)
 tp,tn,fp,fn=(counts[k] for k in ('tp','tn','fp','fn'))
 predicted=sum(pred) if okay else None
 return {'complete':okay,'fixed_denominator':n,'valid_outcomes':valid,'invalid_or_missing_outcomes':n-valid,
  'confusion_counts_on_valid_outcomes':counts,'gold_positive_count':positive,
  'predicted_positive_count':predicted,'predicted_positive_fraction':predicted/n if okay else None,
  'accuracy':(tp+tn)/n if okay else None,'error':(fp+fn)/n if okay else None,
  'balanced_accuracy':(tp/positive+tn/(n-positive))/2 if okay and 0<positive<n else None,
  'f1':(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0) if okay else None,
  'signed_count_bias':predicted-positive if okay else None,
  'absolute_ratio_error':abs(predicted-positive)/n if okay else None,
  'opposing_gold_error_pairs':min(fp,fn) if okay else None,
  'canceled_gold_error_mass':2*min(fp,fn)/n if okay else None,
  'constant_prediction':('Y' if predicted==n else 'N' if predicted==0 else None) if okay else None,
  'count_status':'computed_on_all128' if okay else 'abstain_incomplete_relation'}

def evaluation(pred,reference,gold,reference_only=False):
 fidelity=None if reference_only else disagreement(pred,reference)
 paired=complete(pred) and complete(reference)
 transport=None
 if paired and not reference_only:
  wr=sum(p==g and s!=g for p,s,g in zip(pred,reference,gold))
  rw=sum(p!=g and s==g for p,s,g in zip(pred,reference,gold))
  transport={'reference_wrong_execution_right':wr,'reference_right_execution_wrong':rw,
   'execution_minus_reference_gold_error':(rw-wr)/128,
   'signed_count_change_from_reference':sum(pred)-sum(reference),
   'absolute_ratio_change_from_reference':abs(sum(pred)-sum(reference))/128,
   'candidate_flips':fidelity['numerator'],
   'opposing_flip_pairs':(fidelity['numerator']-abs(sum(pred)-sum(reference)))//2}
 return {'gold':gold_metrics(pred,gold),'singleton_fidelity':fidelity,
  'fidelity_reference':'not_applicable_reference_output' if reference_only else 'fresh_S_same_repeat',
  'gold_and_count_transport_from_reference':transport}

def empirical_indicator(fidelity,tau):
 return None if fidelity is None or not fidelity['complete'] else tau.denominator*fidelity['numerator']<=tau.numerator*fidelity['fixed_denominator']

def arithmetic_break_even(k):
 if k==1:return {'equal_or_lower_call_cost_relations':None,'strictly_lower_call_cost_relations':None}
 saving=128-128//k
 return {'equal_or_lower_call_cost_relations':math.ceil(Fraction(1152,saving)),
  'strictly_lower_call_cost_relations':1152//saving+1,
  'assumption':'same cost and fidelity on future relations; arithmetic only, not measured transfer'}
