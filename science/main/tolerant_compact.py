#!/usr/bin/env python3
"""Conservative compact-label parsing with an explicitly allowed delimiter set."""
import re
from run_validation import parse_compact
def parse_delimited_compact(text,n):
    """Return exactly n Y/N labels, allowing whitespace, commas and a terminal period."""
    if not isinstance(text,str) or type(n) is not int or n<=0:return None
    s=text.strip()
    if s.endswith('.'):s=s[:-1]
    if not s or not re.fullmatch(r'[YN,\s]+',s):return None
    compact=re.sub(r'[,\s]','',s)
    return parse_compact(compact,n)
