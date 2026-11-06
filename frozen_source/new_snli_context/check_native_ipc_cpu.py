#!/usr/bin/env python3
"""Check installed native IPC paths and bind CPU-only AF_UNIX/ZMQ endpoints."""
import ast
import importlib.util
import os
import socket
import sys
from uuid import uuid4
from common import *


def main():
    assert 'torch' not in sys.modules and 'vllm' not in sys.modules
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    utils=(installed/'utils/__init__.py').read_text()
    parsed=ast.parse(utils)
    node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='get_open_zmq_ipc_path')
    implementation=ast.get_source_segment(utils,node)
    assert 'envs.VLLM_RPC_BASE_PATH' in implementation and 'uuid4()' in implementation
    import zmq
    results=[]
    for model in MODELS:
        directory=ROOT/'rpc'/model[0];directory.mkdir(parents=True,exist_ok=True)
        address=directory/str(uuid4());assert not address.exists()
        assert len(str(address).encode('utf8'))<=107
        unix=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:unix.bind(str(address))
        finally:
            unix.close()
            if address.exists():address.unlink()
        zmq_address=directory/str(uuid4());assert not zmq_address.exists()
        context=zmq.Context();channel=context.socket(zmq.PAIR)
        try:channel.bind('ipc://'+str(zmq_address))
        finally:
            channel.close(linger=0);context.term()
            if zmq_address.exists():zmq_address.unlink()
        results.append({'model':model,'VLLM_RPC_BASE_PATH':str(directory),
            'native_uuid_endpoint_path_bytes':len(str(address).encode('utf8')),
            'linux_AF_UNIX_bind':'PASS','pyzmq_ipc_bind':'PASS','own_test_socket_only_removed':True})
    sources=('envs.py','utils/__init__.py','v1/serial_utils.py','v1/engine/core_client.py','v1/engine/core.py')
    write_json(ROOT/'verification/NATIVE_IPC_CPU_CHECK.json',{'status':'PASS','models':results,
        'native_installed_source_sha256':{p:sha(installed/p) for p in sources},
        'get_open_zmq_ipc_path_source':implementation,'native_RPC_variable_only_no_package_patch':True,
        'real_model_judgments':0,'GPU_execution':False,'torch_imported':False})
    assert 'torch' not in sys.modules and 'vllm' not in sys.modules
    print(compact({'status':'PASS','models':results,'GPU_execution':False}),flush=True)


if __name__=='__main__':main()
