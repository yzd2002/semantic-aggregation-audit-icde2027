#!/usr/bin/env python3
"""Order-symmetric combination of two fixed keyed results and an item fallback."""

def combine(first, second, fallback):
    """Return a complete vector or None; only disagreements require fallback."""
    if first is None or second is None or fallback is None:
        return None
    if not (len(first) == len(second) == len(fallback)):
        raise ValueError('All vectors must have the same length.')
    if any(x not in ('Y', 'N') for x in first + second):
        raise ValueError('The keyed vectors must be complete and binary.')
    if any(x not in ('Y', 'N', None) for x in fallback):
        raise ValueError('A fallback label must be binary or missing.')
    result = []
    for a, b, h in zip(first, second, fallback):
        if a == b:
            result.append(a)
        elif h is None:
            return None
        else:
            result.append(h)
    return result


def canonicalize(items):
    """Sort one unchanged chunk by its unique stable local IDs."""
    result = sorted(items, key=lambda x: x['id'])
    if [x['id'] for x in result] != list(range(1, len(result) + 1)):
        raise ValueError('Chunk IDs must be unique and consecutive from one.')
    return result
