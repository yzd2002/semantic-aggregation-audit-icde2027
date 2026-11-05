#!/usr/bin/env python3
"""Fixed ACORD prompts and strict ID-keyed output contracts; no model execution."""
import json,re

SYSTEM=('Assess how well each contract clause addresses the stated drafting query. '
        'Mark Y only when the clause addresses most key substantive requirements, allowing minor omissions. '
        'Mark N when it addresses only some key aspects, is unrelated, or lacks sufficient information. '
        'Use only the provided query and clause text; do not assume omitted terms. Give only the requested labels.')
RETRY_SUFFIX='\nYour previous answer had the wrong format; follow the output instructions exactly.'
IDS=list(range(1,9))

def validate(row):
    if not isinstance(row.get('query'),str) or not row['query'].strip() or len(row.get('items',[]))!=32:
        raise ValueError('Expected nonempty query and a fixed 32-candidate pool')
    if any(not isinstance(x.get('text'),str) or not x['text'].strip() or len(x['text'])>1800 for x in row['items']):
        raise ValueError('Invalid fixed clause prefix')

def keyed_prompt(row,start,reverse=False):
    validate(row)
    if start not in (0,8,16,24):raise ValueError('invalid chunk start')
    items=[{'id':i+1,'clause':x['text']} for i,x in enumerate(row['items'][start:start+8])]
    if reverse:items.reverse()
    text='Drafting query: '+row['query']+'\nClauses in order:\n'+'\n'.join(json.dumps(x,ensure_ascii=False) for x in items)
    text+=('\nReturn only a JSON object mapping each given clause id to Y or N for that clause. '
           'Use the given ids, not positions; include every id once, in increasing numeric order. '
           'Do not add spaces or explanations.')
    return text,[x['id'] for x in items]

def point_prompt(row,index):
    validate(row)
    if type(index) is not int or not 0<=index<32:raise ValueError('invalid candidate index')
    return 'Drafting query: '+row['query']+'\nClause: '+json.dumps({'clause':row['items'][index]['text']},ensure_ascii=False)+'\nAnswer ONLY Y or N.'

def keyed_regex():
    return r'\{'+','.join(re.escape(json.dumps(str(i))+':')+r'"[YN]"' for i in IDS)+r'\}'

def parse_keyed(text):
    if not isinstance(text,str) or re.fullmatch(keyed_regex(),text) is None:return None
    obj=json.loads(text)
    if list(obj)!=[str(i) for i in IDS] or set(obj.values())-{'Y','N'}:return None
    return {int(k):v for k,v in obj.items()}

def parse_point(text):return text if text in ('Y','N') else None
