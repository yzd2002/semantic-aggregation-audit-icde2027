import os
from pathlib import Path
import torch
import torch.distributed as dist
import single_rank_nccl_compat as compat
assert compat.ACTIVE
path=Path('/dev/shm/intentsat-review-20261102/controlled_order')/('nccl_probe_store_'+str(os.getpid()))
torch.cuda.set_device(0)
dist.init_process_group('nccl',init_method='file://'+str(path),rank=0,world_size=1)
group=dist.new_group([0],backend='gloo')
assert dist.get_backend(group)=='nccl' and dist.get_world_size(group)==1 and dist.get_rank(group)==0
dist.barrier(group=group)
try:dist.all_reduce(torch.tensor([1.0]),group=group)
except RuntimeError as e:
    assert 'Unexpected communication' in str(e);print('Expected CPU communication rejection:',str(e),flush=True)
else:raise AssertionError('Unexpected CPU operation was not rejected')
dist.destroy_process_group(group);dist.destroy_process_group()
print('PASS real singleton NCCL group and barrier; non-identity communication fails closed.',flush=True)
