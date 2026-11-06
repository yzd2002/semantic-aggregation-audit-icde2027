"""Persist every actual engine return before any count, prompt, or parser check."""
import os,time
from common import compact

def record_returns(path,jobs,values,stage,block=None):
    actual=[]
    for position,response in enumerate(values):
        prompts=getattr(response,'prompt_token_ids',None)
        prompt_ids=list(prompts) if prompts is not None else None
        outputs=[]
        for output_position,answer in enumerate(getattr(response,'outputs',[]) or []):
            tokens=getattr(answer,'token_ids',None)
            outputs.append({'output_position':output_position,'raw':getattr(answer,'text',None),
                'output_token_ids':list(tokens) if tokens is not None else None,
                'finish_reason':getattr(answer,'finish_reason',None),'stop_reason':getattr(answer,'stop_reason',None)})
        actual.append({'return_position':position,'engine_request_id':getattr(response,'request_id',None),
            'actual_prompt_token_ids':prompt_ids,'actual_input_tokens':len(prompt_ids) if prompt_ids is not None else None,
            'outputs':outputs,'actual_output_tokens':sum(len(o['output_token_ids']) for o in outputs)
            if outputs and all(o['output_token_ids'] is not None for o in outputs) else None})
    row={'stage':stage,'block':block,'returned_unix':time.time(),'expected_batch_request_ids':[j['request_id'] for j in jobs],
        'actual_return_count':len(actual),'actual_returns':actual,
        'journal_written_before_count_identity_or_parser_validation':True}
    # One batch per append, with all actual outputs including any extra/unmatched ones.
    with path.open('a',encoding='utf8') as f:
        f.write(compact(row)+'\n');f.flush();os.fsync(f.fileno())
    return row

def generate_preserving_returns(engine,jobs,parameters,journal_path,stage,block=None):
    values=engine.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in jobs],parameters,use_tqdm=False)
    record_returns(journal_path,jobs,values,stage,block)
    if len(values)!=len(jobs):raise RuntimeError('Engine returned response count mismatch after full raw preservation')
    return values
