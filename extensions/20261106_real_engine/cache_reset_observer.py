"""Observe the real native BlockPool reset; never alter its inputs/result.

vLLM 0.10.2 V1's public forwarding chain drops the underlying bool. The native
pool still has an inspectable bool and hash map. Recording these actual values
is stronger than treating its outer None as success or failure.
"""
import functools
import json
import os
from pathlib import Path
import time


def install():
    from vllm.v1.core.block_pool import BlockPool
    original = BlockPool.reset_prefix_cache
    if getattr(original, "_intent_audit_observer", False):
        return

    @functools.wraps(original)
    def observed(self):
        before = len(self.cached_block_hash_to_block)
        result = original(self)
        after = len(self.cached_block_hash_to_block)
        row = {"pid": os.getpid(), "native_class": type(self).__module__ + "." + type(self).__name__,
               "native_result": result, "cached_hash_count_before": before,
               "cached_hash_count_after": after,
               "num_gpu_blocks": self.num_gpu_blocks,
               "num_free_blocks_after": self.get_num_free_blocks(),
               "observed_unix": time.time(), "result_not_modified": True}
        path = Path(os.environ["INTENTSAT_CACHE_RESET_LOG"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush(); os.fsync(f.fileno())
        return result

    observed._intent_audit_observer = True
    BlockPool.reset_prefix_cache = observed
