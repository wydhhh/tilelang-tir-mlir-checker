"""Snapshot verified inputs and dispatch isolated execution; no server config."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import signal
import base64
from pathlib import Path

from .trace_io import InputError


def request_dynamic(group,case=None,*,dynamic=True,seed=0,timeout=120,max_steps=2_000_000,stages=None,translation=None,
                    tir_backend='ast',tir_python=None,mlir_backend='text',ir_python=None):
    from .trace_io import STAGES
    selected = list(STAGES) if stages is None else list(stages)
    if not selected or len(set(selected)) != len(selected) or any(s not in STAGES for s in selected):
        raise InputError('stages 必须为非空且不重复的完整阶段 ID')
    if tir_backend not in {'ast','tvm'} or (tir_python is not None and tir_backend != 'tvm'):
        raise InputError('tir_backend 必须是 ast/tvm；tir_python 只适用于 tvm')
    if mlir_backend not in {'text','official'}:
        raise InputError('mlir_backend 必须是 text/official')
    if ir_python is not None and tir_backend!='tvm' and mlir_backend!='official':
        raise InputError('ir_python 需要至少一个官方前端')
    if ir_python and tir_python and str(ir_python)!=str(tir_python):
        raise InputError('跨阶段同输入检查使用同一个 Python；ir_python 与 tir_python 不能冲突')
    verify_only = case is None and (tir_backend == 'tvm' or mlir_backend=='official') and not dynamic
    if case is None and not verify_only:
        return {'status':'unsupported','executed':False,
                'reason':'需要显式 --case 测试适配文件，公共引擎不按算子名称猜测输入和参考公式。'}
    case_path=Path(case).expanduser().resolve() if case is not None else None
    if case_path is not None and not case_path.is_file():
        raise InputError(f'测试适配文件不存在：{case}')
    if not 1<=timeout<=3600 or not 100<=max_steps<=20_000_000:
        raise InputError('执行后端、timeout 或 max_steps 参数无效')
    with tempfile.TemporaryDirectory(prefix='tir-mlir-worker-') as directory:
        work=Path(directory)
        trace=work/'trace'
        trace.mkdir()
        stages=[]
        for record in group.stages:
            filename=record['id']+('.tir' if record['kind']=='tir' else '.mlir')
            raw=group.text[record['id']].encode()
            (trace/filename).write_bytes(raw)
            stages.append({'id':record['id'],'file':filename,'sha256':hashlib.sha256(raw).hexdigest(),
                           'bytes':len(raw),'lines':len(raw.splitlines())})
        objects = group.metadata.get('tir_objects',{})
        for entry in objects.values():
            (trace/entry['file']).parent.mkdir(parents=True,exist_ok=True)
            (trace/entry['file']).write_bytes((group.root/entry['file']).read_bytes())
        (trace/'manifest.json').write_text(json.dumps({'schema_version':1,'stages':stages,
            'tir_metadata':group.metadata.get('tir_metadata',{}),'tir_objects':objects,
            'mlir_environment':group.metadata.get('mlir_environment',{})}))
        copied_case=work/'case.py'
        if case_path is not None:
            copied_case.write_bytes(case_path.read_bytes())
        request={'trace':str(trace),'case':str(copied_case) if case_path else None,'original_case':str(case_path),
                 'seed':seed,'dynamic':dynamic,'max_steps':max_steps,'stages':selected,
                 'tir_backend':tir_backend,'mlir_backend':mlir_backend,'verify_only':verify_only}
        if translation is not None:
            source = Path(translation).expanduser().resolve()
            if not source.is_file():
                raise InputError(f'Agent翻译文件不存在：{translation}')
            copied_translation = work/'translation.py'
            copied_translation.write_bytes(source.read_bytes())
            request['translation'] = str(copied_translation)
        request_file,response_file=work/'request.json',work/'response.json'
        request_file.write_text(json.dumps(request))
        env=dict(os.environ)
        package_root=str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH']=package_root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['PYTHONDONTWRITEBYTECODE']='1'
        command=[str(ir_python or tir_python or sys.executable),'-B','-m','tir_mlir_checker.worker',str(request_file),str(response_file)]
        try:
            proc=subprocess.Popen(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                  start_new_session=True)
        except OSError as exc:
            return {'status':'unsupported','executed':False,'reason':f'配置的 Python 不可执行：{exc}',
                    'failure_kind':'python_unavailable'}
        try:
            stdout,stderr=proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.communicate()
            result = {'status':'error','executed':False,'reason':f'隔离执行超过 {timeout}s，子进程已终止',
                      'failure_kind':'timeout'}
        else:
            if proc.returncode!=0 or not response_file.is_file():
                result = {'status':'error','executed':False,'reason':'隔离执行进程异常退出，不能记为算子数值不一致',
                          'worker_returncode':proc.returncode,'stderr':stderr[-12000:]}
            else:
                result=json.loads(response_file.read_text())
            if stderr:
                result['worker_stderr']=stderr[-20000:]
            if stdout:
                result['worker_stdout']=stdout[-20000:]
        result['requested_dynamic']=dynamic
        if case_path is not None:
            result['adapter_sha256']=hashlib.sha256(copied_case.read_bytes()).hexdigest()
        result['input_snapshot']='verified stage text snapshot in isolated temporary directory'
        attachments=result.setdefault('_evidence',{})
        if case_path is not None:
            attachments['adapter.py']=base64.b64encode(copied_case.read_bytes()).decode()
        if translation is not None:
            attachments['translation.py']=base64.b64encode(copied_translation.read_bytes()).decode()
        for record in stages:
            attachments['trace/'+record['file']]=base64.b64encode((trace/record['file']).read_bytes()).decode()
        for entry in objects.values():
            attachments['trace/'+entry['file']]=base64.b64encode((trace/entry['file']).read_bytes()).decode()
        attachments['trace/manifest.json']=base64.b64encode((trace/'manifest.json').read_bytes()).decode()
        return result
