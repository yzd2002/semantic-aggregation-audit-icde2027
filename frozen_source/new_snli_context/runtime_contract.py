"""SNLI runtime metadata adapters; original prepared requests remain untouched."""
from pathlib import Path
import json
import os
import tempfile

from common import compact, digest, read_lines

DIAGNOSTIC_GROUPS=('ordinary_banana','guided_banana','synthetic_format_smoke')


def diagnostic_group(job):
    rid=job['request_id']
    if rid=='diagnostic/ordinary_banana':return 'ordinary_banana'
    if rid.startswith('diagnostic/guided_banana/'):return 'guided_banana'
    if rid.startswith('diagnostic/smoke/'):return 'synthetic_format_smoke'
    raise ValueError('Unrecognized frozen diagnostic ID '+rid)


def load_diagnostic_jobs(bundle,model):
    jobs=read_lines(Path(bundle)/'prepared'/f'{model}_diagnostic_requests.jsonl')
    # Original files have no group field; ordinary unguided BANANA has no
    # grammar hash. Existing guided hashes must match, and are not replaced
    # silently. Prompt, grammar, tokens, request IDs and position stay fixed.
    for job in jobs:
        if 'grammar_sha256' in job and job['grammar_sha256']!=(digest(job['grammar']) if job.get('grammar') is not None else None):
            raise ValueError('Frozen diagnostic grammar hash differs')
    return [{**job,'diagnostic_group':diagnostic_group(job),
        'grammar_sha256':digest(job['grammar']) if job.get('grammar') is not None else None}
        for job in jobs]


def atomic_json(path,value):
    """Durable replacement for mutable runtime status/accounting, never inputs."""
    path=Path(path)
    encoded=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',prefix='.'+path.name+'.',
            suffix='.tmp',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def cache_environment(root,tag,env=None):
    env=dict(os.environ if env is None else env)
    cache=Path(root)/'runtime_cache'/tag
    paths={'TMPDIR':'tmp','XDG_CACHE_HOME':'xdg','HF_HOME':'hf','HF_MODULES_CACHE':'hf_modules',
        'TORCH_HOME':'torch','VLLM_CACHE_ROOT':'vllm','VLLM_CONFIG_ROOT':'vllm_config',
        'NUMBA_CACHE_DIR':'numba','TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton',
        'CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'extensions','FLASHINFER_WORKSPACE_BASE':'flashinfer-base'}
    for name,relative in paths.items():
        path=cache/relative;path.mkdir(parents=True,exist_ok=True);env[name]=str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    return env
