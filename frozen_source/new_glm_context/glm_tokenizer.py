"""Exact local GLM tokenizer and byte-based offsets for its slow tokenizer.

This module loads only tokenizer code/data. It never loads a model or tensors.
The original statistical/semantic helpers remain in common.py unchanged.
"""
from pathlib import Path
from common import compact, digest, prompt, requested_ids, output_ids, grammar

MODEL = Path('/smoothcloud/models/glm-4-9b-chat')


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True,
                                         local_files_only=True, use_fast=False)


def token_bytes(tokenizer, token_id):
    if token_id in tokenizer.added_tokens_decoder:
        return tokenizer.added_tokens_decoder[token_id].content.encode('utf8')
    value = tokenizer.decoder[token_id]
    assert isinstance(value, bytes)
    return value


def prepare(tokenizer, task, records, scope):
    system, user, target_line = prompt(task, records, scope)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    wrapped = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(wrapped, add_special_tokens=False, truncation=False)
    assert ids == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    start = wrapped.index(user) + len('Records:\n')
    end = start + len(target_line)
    start_byte, end_byte = len(wrapped[:start].encode()), len(wrapped[:end].encode())
    chunks = [token_bytes(tokenizer, token_id) for token_id in ids]
    assert b''.join(chunks) == wrapped.encode('utf8'), 'Token bytes do not reproduce the complete prompt'
    offsets = []
    cursor = 0
    for chunk in chunks:
        offsets.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)
    prefix_n = sum(b <= end_byte for a, b in offsets)
    prefix = wrapped[:end]
    return {'system': system, 'unwrapped_user': user, 'wrapped_prompt': wrapped, 'prompt_token_ids': ids,
            'prompt_tokens_preflight': len(ids), 'requested_ids': requested_ids(scope),
            'output_ids': output_ids(scope), 'grammar': grammar(output_ids(scope)),
            'prompt_sha256': digest(wrapped), 'grammar_sha256': digest(grammar(output_ids(scope))),
            'target_input_span_chars': [start, end], 'target_input_span_bytes': [start_byte, end_byte],
            'target_input_line': target_line, 'target_input_prefix': prefix,
            'target_input_prefix_sha256': digest(prefix), 'target_prefix_token_count': prefix_n,
            'target_prefix_token_ids_sha256': digest(compact(ids[:prefix_n])),
            'target_token_end_exact_boundary': bool(prefix_n and offsets[prefix_n-1][1] == end_byte),
            'token_offset_method': 'Exact concatenation of native tiktoken byte pieces and literal added special tokens; no text truncation',
            'token_byte_offsets_sha256': digest(compact(offsets)),
            'fixed_output_prefix_before_target': '{"1":"' if scope != 'full-last' else None}
