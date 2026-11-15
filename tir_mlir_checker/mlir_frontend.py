"""Optional official MLIR parser/verifier. Never permits unregistered dialects.

Operation objects, not their printed assembly, feed the separate simulator.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from .trace_io import STAGES

DIALECTS = ('arith', 'memref', 'scf', 'linalg', 'math', 'func')


def identity():
    import mlir.ir as ir
    import mlir._mlir_libs._mlir as extension
    result = {}
    for name,module in (('python',ir),('extension',extension)):
        path = Path(module.__file__).resolve()
        result[name+'_module'] = str(path)
        result[name+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for filename in ('libMLIRPythonCAPI.so','libMLIRPythonCAPI.dylib','MLIRPythonCAPI.dll'):
        path=Path(extension.__file__).resolve().parent/filename
        if path.is_file():
            result['capi_library']=str(path.resolve())
            result['capi_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            break
    return result


def load_stages(group):
    loaded = {}
    result = {'status':'passed','runs':[],
              'scope':'官方解析和注册操作的 verifier；不证明跨阶段数值等价、内存安全或真机正确。'}
    try:
        import mlir.ir as ir
        for name in DIALECTS: importlib.import_module('mlir.dialects.'+name)
        result['environment'] = identity()
    except (ImportError,OSError) as exc:
        result.update(status='unsupported',reason=f'匹配的 MLIR Python 接口不可用：{exc}')
        result['runs'] = [{'stage':stage,'status':'unsupported','failure_kind':'dependency_unavailable',
                           'diagnostic':str(exc)} for stage in STAGES[3:]]
        return loaded,result
    expected = group.metadata.get('mlir_environment',{})
    result['capture_identity_available']=bool(expected.get('python_sha256') or expected.get('extension_sha256'))
    mismatch = any(expected.get(k) and expected[k] != result['environment'].get(k)
                   for k in ('python_sha256','extension_sha256','capi_sha256'))
    for stage in STAGES[3:]:
        text = group.text[stage]
        record = {'stage':stage,'status':'unsupported','loader':'mlir.ir.Module.parse',
                  'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'diagnostics':[]}
        result['runs'].append(record)
        if mismatch:
            record.update(failure_kind='version_incompatible',diagnostic='MLIR Python/扩展指纹与采集环境不同；未静默回退文本执行。')
            continue
        ctx = ir.Context()
        def diagnostic(diag):
            record['diagnostics'].append({'message':str(diag),'location':str(diag.location),
                                         'severity':str(diag.severity)})
            return True
        with ctx:
            ctx.allow_unregistered_dialects = False
            ctx.load_all_available_dialects()
            handler = ctx.attach_diagnostic_handler(diagnostic)
            try:
                module = ir.Module.parse(text)
                valid = bool(module.operation.verify())
                record.update(status='passed' if valid else 'invalid_ir',verified=valid,
                              registered_dialects_only=True)
                if valid: loaded[stage] = (ctx,module)
            except Exception as exc:
                message = '\n'.join([str(exc)]+[d['message'] for d in record['diagnostics']])
                registration_gap = any(s in message.lower() for s in
                    ('unregistered','unknown dialect','not found for custom op','not registered'))
                record.update(status='unsupported' if registration_gap else 'invalid_ir',
                    failure_kind='dialect_incompatible' if registration_gap else 'parse_or_verify_rejected',
                    diagnostic=f'{type(exc).__name__}: {exc}')
            finally:
                handler.detach()
    states = {r['status'] for r in result['runs']}
    result['status'] = 'error' if 'invalid_ir' in states else ('passed' if states=={'passed'} else 'unsupported')
    return loaded,result


def verify_only(request):
    from .trace_io import load_trace
    group = load_trace(request['trace'])
    result = {'status':'passed','executed':False}
    if request.get('tir_backend') == 'tvm':
        from .tvm_frontend import load_stages as load_tir
        _, result['tir_verification'] = load_tir(group)
    if request.get('mlir_backend') == 'official':
        _, result['mlir_python_verification'] = load_stages(group)
    states = {r['status'] for k,r in result.items() if k.endswith('verification')}
    result['status'] = 'error' if 'error' in states else ('unsupported' if 'unsupported' in states else 'passed')
    return result
