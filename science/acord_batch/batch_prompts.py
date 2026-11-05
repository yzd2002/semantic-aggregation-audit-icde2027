#!/usr/bin/env python3
"""ACORD fixed-pool, chunk-local-ID prompts for 4/8 item plans."""
import json,re
from acord_prompts import SYSTEM,RETRY_SUFFIX,validate

def prompt(row,start,size,reverse=False):
    validate(row)
    if size not in (4,8) or type(start) is not int or start not in range(0,32,size):raise ValueError('Invalid chunk')
    items=[{'id':i+1,'clause':x['text']} for i,x in enumerate(row['items'][start:start+size])]
    if reverse:items.reverse()
    text='Drafting query: '+row['query']+'\nClauses in order:\n'+'\n'.join(json.dumps(x,ensure_ascii=False) for x in items)
    text+=('\nReturn only a JSON object mapping each given clause id to Y or N for that clause. '
           'Use the given ids, not positions; include every id once, in increasing numeric order. '
           'Do not add spaces or explanations.')
    return text,[x['id'] for x in items]

def regex(size):
    if size not in (4,8):raise ValueError('Invalid size')
    return r'\{'+','.join(re.escape(json.dumps(str(i))+':')+r'"[YN]"' for i in range(1,size+1))+r'\}'

def parse(text,size):
    if not isinstance(text,str) or re.fullmatch(regex(size),text) is None:return None
    value=json.loads(text)
    if list(value)!=[str(i) for i in range(1,size+1)] or set(value.values())-{'Y','N'}:return None
    return [value[str(i)] for i in range(1,size+1)]
