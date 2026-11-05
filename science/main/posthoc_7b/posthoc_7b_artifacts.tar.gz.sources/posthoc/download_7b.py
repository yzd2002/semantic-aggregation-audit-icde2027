#!/usr/bin/env python3
"""Download pinned public model assets into tmpfs and verify official Git/LFS hashes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,requests,threading,time
base=Path('/dev/shm/intentsat-7b-followup-20261031')
dest=base/'model/Qwen2.5-7B-Instruct'
rev='a09a35458c702b33eeacc393d103063234e8bc28'
official=f'https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct/revision/{rev}?blobs=true'
mirror=f'https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/{rev}/'
small={
 'LICENSE':(11343,'6634c8cc3133b3848ec74b9f275acaaa1ea618ab'),
 'README.md':(6240,'4e9536cebaeb5cac673952ad5d0eeae469900f81'),
 'config.json':(663,'0178295f88afc3c7f279ed284f961f8c1be00654'),
 'generation_config.json':(243,'0eb3c536657dcd12626e09eca4b6198c0cbcde1e'),
 'merges.txt':(1671839,'20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0'),
 'model.safetensors.index.json':(27752,'14d037fdda5a1311dcc0275003a7a370d84114e6'),
 'tokenizer.json':(7031645,'443909a61d429dff23010e5bddd28ff530edda00'),
 'tokenizer_config.json':(7305,'07bfe0640cb5a0037f9322287fbfc682806cf672'),
 'vocab.json':(2776833,'4783fe10ac3adce15ac8f358ef5462739852c569')}
weights={
 'model-00001-of-00004.safetensors':(3945441440,'a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7'),
 'model-00002-of-00004.safetensors':(3864726352,'f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185'),
 'model-00003-of-00004.safetensors':(3864726424,'8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5'),
 'model-00004-of-00004.safetensors':(3556377672,'1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd')}
manifest={'official_revision_metadata_url':official,'mirror_resolve_base':mirror,'revision':rev,'small_git_blob_sha1':small,'weights_lfs_sha256':weights}
manifest_path=base/'MODEL_OFFICIAL_EXPECTED.json'
with manifest_path.open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2);f.write('\n')
st=os.statvfs('/dev/shm');free=st.f_bavail*st.f_frsize
need=sum(x[0] for x in small.values())+sum(x[0] for x in weights.values())
if free<need+4*1024**3:raise RuntimeError('Insufficient tmpfs headroom for model')
print('Starting verified download, bytes:',need,'tmpfs free:',free,flush=True)
s=requests.Session();s.trust_env=False
for name,(size,blob) in small.items():
 response=s.get(mirror+name,timeout=(20,120));response.raise_for_status();content=response.content
 git=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
 if len(content)!=size or git!=blob:raise RuntimeError('Official Git blob hash mismatch: '+name)
 with (dest/name).open('xb') as f:f.write(content)
 print('Metadata verified:',name,flush=True)
stop=threading.Event()
def download(item):
 name,(size,want)=item;path=dest/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 pos=0;reported=0;start=time.monotonic()
 try:
  for attempt in range(4):
   if stop.is_set():raise RuntimeError('Another shard failed')
   try:
    session=requests.Session();session.trust_env=False
    with session.get(mirror+name,headers={'Range':f'bytes={pos}-{size-1}'},stream=True,timeout=(20,120)) as response:
     if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {pos}-{size-1}/{size}':raise RuntimeError('Unexpected range response')
     for block in response.iter_content(chunk_size=1024*1024):
      if stop.is_set():raise RuntimeError('Another shard failed')
      if not block:continue
      if os.pwrite(fd,block,pos)!=len(block):raise RuntimeError('Short tmpfs write')
      pos+=len(block)
      if pos-reported>=512*1024*1024:
       st=os.statvfs('/dev/shm')
       if st.f_bavail*st.f_frsize<2*1024**3:raise RuntimeError('Tmpfs safety reserve reached')
       print(name,'received MiB',pos//1024**2,'/',size//1024**2,flush=True);reported=pos
    if pos==size:break
    raise RuntimeError('Incomplete response')
   except Exception as exc:
    print(name,'retry',attempt+1,'after',type(exc).__name__,flush=True)
    if attempt==3:stop.set();raise
    time.sleep(1)
  os.fsync(fd)
 finally:os.close(fd)
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if path.stat().st_size!=size or h.hexdigest()!=want:stop.set();raise RuntimeError('Official LFS SHA-256 mismatch: '+name)
 print(name,'verified in',round(time.monotonic()-start,1),'seconds',flush=True)
 return name
with ThreadPoolExecutor(max_workers=4) as pool:
 for completed in pool.map(download,weights.items()):print('Completed:',completed,flush=True)
with (base/'MODEL_LOCAL_SHA256.txt').open('x',encoding='utf8') as out:
 for p in sorted(dest.iterdir()):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  out.write(h.hexdigest()+'  '+p.name+'\n')
print('All model files verified and local SHA-256 manifest saved.',flush=True)
