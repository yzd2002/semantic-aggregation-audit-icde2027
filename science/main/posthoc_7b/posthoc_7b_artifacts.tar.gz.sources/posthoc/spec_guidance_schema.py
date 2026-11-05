#!/usr/bin/env python3
"""The post-hoc specification output schema; semantic correctness is not implied."""
SPEC_SCHEMA = {
    'type': 'object',
    'properties': {
        'product_type': {'type': 'string', 'minLength': 1},
        'required': {'type': 'array', 'items': {'type': 'string'}},
        'preferred': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['product_type', 'required', 'preferred'],
    'additionalProperties': False,
}
