import os,time,traceback
from pathlib import Path
import torch.distributed as dist
path=Path('/dev/shm/intentsat-review-20261102/controlled_order')/('gloo_probe_store_'+str(os.getpid()))
print('GLOO_SOCKET_IFNAME',os.environ.get('GLOO_SOCKET_IFNAME'),'GLOO_DEVICE_TRANSPORT',os.environ.get('GLOO_DEVICE_TRANSPORT'),flush=True)
try:
    dist.init_process_group('gloo',init_method='file://'+str(path),rank=0,world_size=1)
    print('CPU GLOO PASS',flush=True);dist.destroy_process_group()
except Exception:
    traceback.print_exc();raise
