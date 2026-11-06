import os
if os.environ.get('INTENTSAT_SINGLE_RANK_NCCL_COMPAT')=='1':
    import single_rank_nccl_compat
    single_rank_nccl_compat.install()
