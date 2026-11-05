#!/usr/bin/env python3
"""Independently reparse the saved synthetic vLLM dependency smoke check."""
import argparse
import json
import re
from pathlib import Path

SYSTEM = ("You judge whether a product fully matches a shopper's query. "
          "Mark Y only if it matches the requested product type and every stated essential modifier; otherwise mark N. "
          "Use only the supplied information. Do not assume unknown product properties.")
ITEMS = ['Red ceramic mug with a handle.', 'Blue ceramic mug.', 'Red plastic mug.',
         'Red ceramic plate.', 'Red ceramic coffee mug.', 'Set of red ceramic bowls.',
         'Red mug lid only.', 'Small red ceramic cup.']


def compact(text, n):
    return re.fullmatch('[YN]{' + str(n) + '}', text) is not None


def keyed(text):
    if re.fullmatch(r'\{' + ','.join('"' + str(i) + '":"[YN]"' for i in range(1, 9)) + r'\}', text) is None:
        return False
    obj = json.loads(text)
    return list(obj) == [str(i) for i in range(1, 9)] and all(x in ('Y', 'N') for x in obj.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--model', help='Optional local tokenizer path; no weights or GPU are used')
    args = ap.parse_args()
    setup = json.loads((args.run / 'setup.json').read_text())
    records = json.loads((args.run / 'records.json').read_text())
    controls = json.loads((args.run / 'controls.json').read_text())
    summary = json.loads((args.run / 'summary.json').read_text())
    assert setup['protocol'] == 'PROTOCOL_vllm_declared_deps_smoke.md'
    assert setup['model_directory_name'] == 'Qwen3-4B-Instruct-2507'
    assert setup['versions'] == {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                                  'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11',
                                  'setuptools': '79.0.1'}
    assert setup['engine'] == 'V1' and setup['dtype'] == 'bfloat16' and setup['max_model_len'] == 4096
    assert setup['max_num_seqs'] == 16 and setup['gpu_memory_utilization'] == .70
    assert setup['enforce_eager'] and setup['prefix_caching']
    assert [x['case'] for x in records] == ['control_unconstrained', 'control_compact', 'control_keyed',
                                             'synthetic_compact', 'synthetic_keyed']
    first, c4, k8, c8, product_keyed = records
    regex_keyed = r'\{' + ','.join('"' + str(i) + '":"[YN]"' for i in range(1, 9)) + r'\}'
    for item, regex in zip(records, (None, '[YN]{4}', regex_keyed, '[YN]{8}', regex_keyed)):
        assert item['regex'] == regex and isinstance(item['output'], str)
        assert item['finish_reason'] in ('stop', 'length') and item['elapsed_seconds'] >= 0
        assert item['prompt_tokens'] > 0 and item['output_tokens'] > 0
    assert all(x['system'] == 'Follow the user instruction exactly.' and
               x['prompt'] == 'Reply with only the word BANANA.' for x in records[:3])
    products = '\n'.join(json.dumps({'id': i + 1, 'text': x}, ensure_ascii=False) for i, x in enumerate(ITEMS))
    prefix = 'Query: red ceramic mug\nProducts in order:\n' + products
    assert c8['system'] == product_keyed['system'] == SYSTEM
    assert c8['prompt'] == prefix + ('\nReturn exactly 8 characters, one Y or N per product in the listed order. '
                                     'No spaces, punctuation, or explanations.')
    assert product_keyed['prompt'] == prefix + ('\nReturn only a JSON object mapping each given product id to Y or N '
        'for that product. Use the given ids, not positions; include every id once, in increasing numeric order. '
        'Do not add spaces or explanations.')
    expected_controls = {'unconstrained_invalid_compact': not compact(first['output'], 4),
                         'unconstrained_invalid_keyed': not keyed(first['output']),
                         'compact_valid': compact(c4['output'], 4), 'keyed_valid': keyed(k8['output'])}
    expected_synthetic = {'compact_valid': compact(c8['output'], 8), 'keyed_valid': keyed(product_keyed['output'])}
    assert controls == expected_controls == summary['controls'] and all(controls.values())
    assert expected_synthetic == summary['synthetic'] and all(expected_synthetic.values())
    assert summary['gpu'] == 'NVIDIA GeForce RTX 5090' and summary['cuda'] == '12.8'
    if args.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        for item in records:
            messages = [{'role': 'system', 'content': item['system']}, {'role': 'user', 'content': item['prompt']}]
            count = len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            assert count == item['prompt_tokens'], (item['case'], count, item['prompt_tokens'])
    print('Synthetic prompts, controls, strict parsers, recorded versions and result summaries verified.')
    print('Tokenizer prompt lengths checked:', bool(args.model))


if __name__ == '__main__':
    main()
