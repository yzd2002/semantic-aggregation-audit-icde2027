#!/usr/bin/env python3
"""Exercise accepted delimiters and reject incomplete, extra and ambiguous labels."""
from tolerant_compact import parse_delimited_compact as parse
assert parse('Y,N, N, Y.',4)==list('YNNY')
assert parse('Y N\tN\nY',4)==list('YNNY')
assert parse('YNNY',4)==list('YNNY')
for s,n in [('YNNY.',8),('YNNY..',4),('Y/N',2),('Yes No',2),('YNNY explanation',4),('Y N',3),('Y N Y',2),('',1)]:
    assert parse(s,n) is None,(s,n)
assert parse('Y',0) is None and parse('Y',True) is None
print('Delimiter-tolerant parser checks passed.')
