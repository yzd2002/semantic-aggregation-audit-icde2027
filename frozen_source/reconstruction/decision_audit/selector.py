"""Pure development-only empirical selectors; no source/gold/test-data access."""
from fractions import Fraction
from common import KS,GROUPS,config_id
TOLERANCES=(('0.05',Fraction(1,20)),('0.10',Fraction(1,10)),('0.20',Fraction(1,5)))

def select_policies(vectors,n=128,tau=Fraction(1,20)):
 assert tau in [value for name,value in TOLERANCES]
 references=vectors.get('singleton',{})
 reference_valid=all(len(references.get(r,[]))==n and all(type(v) is int and v in (0,1) for v in references.get(r,[])) for r in (1,2))
 result={}
 for selector,types,observations in [('Input',('C','I'),2),('Output',('C','O'),2),('Repeat',('C',),4)]:
  ledger=[]
  for k in KS:
   for group in GROUPS:
    for kind in types:
     identifier=config_id(k,group,kind);reps=vectors.get(identifier,{})
     complete=all(len(reps.get(r,[]))==n and all(type(v) is int and v in (0,1) for v in reps.get(r,[])) for r in range(1,observations+1))
     errors=sum(int(pred!=ref) for r in range(1,observations+1) for pred,ref in zip(reps[r],references[(r-1)%2+1])) if complete and reference_valid else None
     denominator=n*observations;eligible=errors is not None and tau.denominator*errors<=tau.numerator*denominator
     ledger.append({'configuration_id':identifier,'k':k,'grouping':group,'execution_type':kind,'observations':observations,
      'complete_candidate':complete,'complete_reference':reference_valid,'disagreements':errors,'fixed_denominator':denominator,
      'raw_disagreement':float(Fraction(errors,denominator)) if errors is not None else None,'eligible':eligible,
      'one_execution_calls':n//k})
  valid=[x for x in ledger if x['eligible']]
  if valid:
   best=min(valid,key=lambda x:(x['one_execution_calls'],Fraction(x['disagreements'],x['fixed_denominator']),GROUPS.index(x['grouping']),0 if x['execution_type']=='C' else 1))
   chosen={'configuration_id':best['configuration_id'],'k':best['k'],'grouping':best['grouping'],'execution_type':best['execution_type'],
    'one_execution_calls':best['one_execution_calls'],'reason':'eligible_min_calls_then_exact_loss_then_fixed_tie'}
  else:chosen={'configuration_id':'singleton','k':1,'grouping':'none','execution_type':'S','one_execution_calls':n,
   'reason':'reference_calibration_unavailable' if not reference_valid else 'no_eligible_batch'}
  result[selector]={'chosen':chosen,'reference_calibration_available':reference_valid,'tolerance_numerator':tau.numerator,'tolerance_denominator':tau.denominator,
   'standalone_calibration_calls':1152,'standalone_calibration_requested_items':3328,'eligibility_ledger':ledger}
 return result

def select_all_tolerances(vectors,n=128):
 return {name:select_policies(vectors,n,tau) for name,tau in TOLERANCES}
