#!/usr/bin/env python3
"""Copy only public tokenizer/config assets into this experiment directory."""
from common import *
cpu_environment()
import shutil

ASSETS=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
 'special_tokens_map.json','vocab.json','merges.txt','tokenizer.model','tokenization_chatglm.py',
 'configuration_chatglm.py','README.md','README_en.md','LICENSE')

def main():
 locations=json.loads((ROOT/'model_locations.json').read_text())
 root=ROOT/'tokenizers';root.mkdir(exist_ok=False);report={}
 for model in MODELS:
  source=Path(locations[model]);dest=root/model;dest.mkdir();files={}
  for name in ASSETS:
   p=source/name
   if not p.is_file():continue
   shutil.copyfile(p,dest/name)
   assert sha(p)==sha(dest/name)
   files[name]={'sha256':sha(p),'bytes':p.stat().st_size}
  report[model]={'model_path':str(source),'files':files}
 write_json(ROOT/'TOKENIZER_ASSETS.json',report)
 tokens=load_tokenizers()
 sample={'features':{'premise':'A person is walking outside.','hypothesis':'Someone is outdoors.'}}
 for model,tok in tokens.items():
  p=prepared_prompt(tok,'snli',[sample],'singleton')
  print(model,'tokenizer_only',tok.__class__.__name__,'singleton_tokens',p['prompt_tokens_preflight'],flush=True)

if __name__=='__main__':main()
