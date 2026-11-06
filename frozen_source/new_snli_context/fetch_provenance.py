#!/usr/bin/env python3
"""Public-source retrieval with TLS verification and no proxy/authentication."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,ssl,time,urllib.request,urllib.parse

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT/'sources'
URLS={
 'stanford_snli.html':'https://nlp.stanford.edu/projects/snli/',
 'acl_D15-1075.html':'https://aclanthology.org/D15-1075/',
 'cc_by_sa_4.0.html':'https://creativecommons.org/licenses/by-sa/4.0/',
 'hf_api_snli.json':'https://huggingface.co/api/datasets/stanfordnlp/snli',
 'tfds_snli_checksums.txt':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/snli.txt',
}

def fetch(item):
 name,url=item
 result={'source_url':url,'filename':name,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'transport':'urllib HTTPS, ssl.create_default_context(), ProxyHandler({}), no authentication'}
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
  with opener.open(urllib.request.Request(url,headers={'User-Agent':'IntentSat2S research provenance audit/20261106'}),timeout=35) as response:
   data=response.read()
   u=urllib.parse.urlsplit(response.url)
   final_url=urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,'',''))
   result.update(status='ok',http_status=response.status,final_url_query_omitted=final_url,
                 bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  (SOURCES/name).write_bytes(data)
 except Exception as exc:
  result.update(status='error',error_type=type(exc).__name__,error=str(exc))
 return result

if __name__=='__main__':
 SOURCES.mkdir(exist_ok=True)
 with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,URLS.items()))
 (SOURCES/'provenance_retrieval.json').write_text(json.dumps(results,indent=2)+'\n')
 for row in results:print(json.dumps(row),flush=True)
