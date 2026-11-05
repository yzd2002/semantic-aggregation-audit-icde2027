#!/usr/bin/env python3
"""Download indexed public artifacts anonymously, verifying every final byte."""
import argparse,hashlib,json,os,time,urllib.request,urllib.error,urllib.parse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--index',type=Path,default=Path(__file__).resolve().parents[1]/'asset_index.json');p.add_argument('--direct',action='store_true',help='Use direct HTTPS instead of inherited proxy settings; certificate validation stays enabled');a=p.parse_args()
index=json.loads(a.index.read_text());assert not a.out.exists(),'Use a new output directory';a.out.mkdir(parents=True);attempt_dir=a.out/'.download_attempts';attempt_dir.mkdir();receipts=[]
for item in index['assets']:
 name=item['name'];assert Path(name).name==name and name not in ('','.','..')
 routes=[('browser',item['download_url'])]
 if item.get('api_url'):routes.append(('public_api',item['api_url']))
 errors=[];success=None
 for route,url in routes:
  u=urllib.parse.urlsplit(url);assert u.scheme=='https' and not u.query and not u.fragment
  assert (route=='browser' and u.netloc=='github.com') or (route=='public_api' and u.netloc=='api.github.com' and u.path.startswith('/repos/'+index['repository']+'/releases/assets/'))
  for attempt in (1,2):
   temporary=attempt_dir/(name+'.'+route+'.'+str(attempt));h=hashlib.sha256();total=0
   req=urllib.request.Request(url,headers={'User-Agent':'Semantic-Aggregation-Public-Reproduction','Accept':'application/octet-stream'})
   assert not any(k.lower() in ('authorization','cookie') for k,v in req.header_items())
   op=urllib.request.build_opener(urllib.request.ProxyHandler({})) if a.direct else urllib.request.build_opener()
   try:
    with op.open(req,timeout=30) as response,temporary.open('xb') as stream:
     assert response.status==200
     for block in iter(lambda:response.read(1<<20),b''):
      total+=len(block);assert total<=item['bytes'];h.update(block);stream.write(block)
     stream.flush();os.fsync(stream.fileno())
    assert total==item['bytes'] and h.hexdigest()==item['sha256'],'Artifact hash/length mismatch'
    temporary.rename(a.out/name);success={'name':name,'bytes':total,'sha256':h.hexdigest(),'anonymous':True,'route':route,'prior_errors':errors};break
   except Exception as error:
    errors.append({'route':route,'attempt':attempt,'error_type':type(error).__name__})
    if isinstance(error,urllib.error.HTTPError) and error.code==429:raise
    time.sleep(attempt)
  if success:break
 if not success:raise RuntimeError('Public download failed for '+name+'; failed attempts are retained in .download_attempts')
 receipts.append(success);print('PASS',name,flush=True)
(a.out/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['name']+'\n' for x in receipts))
(attempt_dir/'DOWNLOAD_RECEIPT.json').write_text(json.dumps({'status':'PASS','files':receipts,'TLS_certificate_validation':True},indent=2)+'\n')
print('PASS all',len(receipts),'public artifact downloads; no model inference')
