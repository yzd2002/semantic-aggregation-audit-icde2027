"""Process-local rank-one group transport adaptation, with communication guards.

No sysfs interception, fake process groups, mocked collective result, or package
file mutation. Only new_group(ranks=[0], backend='gloo') in an already initialized
world-size-one NCCL world is redirected to an actual singleton NCCL group.
"""
import functools,inspect,json,os
from pathlib import Path
import torch.distributed as dist

ACTIVE=False
GROUPS=set()
ROOT=Path('/dev/shm/intentsat-review-20261102/controlled_order/runtime_override')
def event(kind,**kwargs):
    record={'event':kind,'pid':os.getpid(),**kwargs}
    with (ROOT/f'events_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print('SINGLE_RANK_NCCL_COMPAT',json.dumps(record),flush=True)
def install():
    global ACTIVE
    if ACTIVE:return
    original=dist.new_group;sig=inspect.signature(original)
    @functools.wraps(original)
    def new_group(*args,**kwargs):
        call=sig.bind_partial(*args,**kwargs)
        if call.arguments.get('backend')=='gloo':
            ranks=call.arguments.get('ranks')
            if (ranks!=[0] or not dist.is_initialized() or dist.get_world_size()!=1 or dist.get_rank()!=0 or dist.get_backend()!='nccl'):
                event('REJECT_GROUP',ranks=ranks);raise RuntimeError('Compatibility adapter permits only ranks=[0] inside a one-rank NCCL world')
            call.arguments['backend']='nccl';group=original(*call.args,**call.kwargs)
            if dist.get_world_size(group)!=1 or dist.get_backend(group)!='nccl':raise RuntimeError('Unexpected replacement process group')
            GROUPS.add(group);event('ACTUAL_SINGLETON_NCCL_GROUP',ranks=ranks,world_size=1)
            return group
        return original(*args,**kwargs)
    dist.new_group=new_group
    # No collective result is fabricated. The inspected vLLM rank-one path
    # bypasses these operations. If it unexpectedly calls one, fail visibly.
    names=('all_reduce','reduce','broadcast','all_gather','all_gather_into_tensor','gather','scatter',
        'reduce_scatter','reduce_scatter_tensor','all_to_all','all_to_all_single','send','recv','isend','irecv',
        'broadcast_object_list','all_gather_object','gather_object','scatter_object_list')
    for name in names:
        operation=getattr(dist,name,None)
        if operation is None:continue
        signature=inspect.signature(operation)
        def make_guard(fn,signature,name):
            @functools.wraps(fn)
            def guarded(*args,**kwargs):
                call=signature.bind_partial(*args,**kwargs);group=call.arguments.get('group')
                if group in GROUPS:
                    event('REJECT_COMMUNICATION',operation=name)
                    raise RuntimeError('Unexpected communication on rank-one CPU-group replacement: '+name)
                return fn(*args,**kwargs)
            return guarded
        setattr(dist,name,make_guard(operation,signature,name))
    ACTIVE=True;event('INSTALLED',guarded_operations=list(names),allowed_operations='real NCCL barrier, group metadata and destruction')
