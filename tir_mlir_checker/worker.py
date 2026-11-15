"""Isolated case loading and bounded interpretation."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import itertools
import json
import platform
import re
import runpy
import sys
import tempfile
import traceback
from pathlib import Path


def execute(request):
    import numpy as np
    from .cases import compare,validate_case,cases_fingerprint
    from .execution import Budget,SemanticFault,Unsupported,make_buffers,provenance
    from .diagnostics import finding
    from .mlir_interpreter import MLIRInterpreter
    from .semantics import compare_symbolic,coverage
    from .tir_interpreter import TIRInterpreter
    from .trace_io import STAGES,load_trace

    result={'status':'passed','executed':False,'backend':'interpreter',
            'seed':request['seed'],'tests':[],'symbolic':[],'stage_support':{},
            'diagnostics':[],'findings':[],'cases':[],'_evidence':{},
            'scope':'当前有限测试集/形状，不是任意输入和物理并行行为的等价证明。',
            'host':{'machine':platform.machine(),'python':sys.version,'numpy':np.__version__}}
    group=load_trace(request['trace'])
    official_functions = {}
    if request.get('tir_backend') == 'tvm':
        from .tvm_frontend import load_stages
        official_functions, result['tir_verification'] = load_stages(group)
    official_mlir = {}
    if request.get('mlir_backend') == 'official':
        from .mlir_frontend import load_stages as load_mlir
        official_mlir,result['mlir_python_verification'] = load_mlir(group)
    case_path=Path(request['case'])
    result['adapter']={'file':request['original_case'],'sha256':hashlib.sha256(case_path.read_bytes()).hexdigest()}
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
        namespace=runpy.run_path(str(case_path),run_name='checker_case_adapter')
        if not callable(namespace.get('make_cases')):
            raise Unsupported('测试适配文件必须定义 make_cases(seed)')
        cases=list(itertools.islice(namespace['make_cases'](request['seed']),33))
    if not 1 <= len(cases) <= 32:
        raise Unsupported('每次要求 1–32 个测试用例，更多用例请分批执行')
    if captured.getvalue():
        result['adapter_log']=captured.getvalue()[-12000:]
    seen=set()
    for case in cases:
        validate_case(case)
        if case.name in seen:
            raise Unsupported('用例名称重复')
        seen.add(case.name)
        result['cases'].append({'name':case.name,'outputs':list(case.outputs),'inouts':list(case.inouts),'symbols':case.symbols,
                                'arguments':[{'shape':list(a.shape),'dtype':str(a.dtype)} for a in case.arguments]})
    selected=request.get('stages',list(STAGES))
    result['cases_fingerprint'] = cases_fingerprint(cases)
    result['requested_stages']=selected
    sources={stage:group.text[stage] for stage in selected}
    def unsupported(stage, exc, method='preflight', case=None):
        line = re.search(r'\bL(\d+)', str(exc))
        location=getattr(exc,'mlir_location',None)
        result['findings'].append(finding('CAP-EXECUTION', 'unsupported', method, str(exc),
            stage=stage, line=location.get('stage_text_line') if location else (int(line[1]) if line and int(line[1]) else None),
            details={'case':case,**({'mlir_location':location} if location else {})}))

    def fault(stage, case, exc, symbolic):
        status = 'uncertain' if symbolic else 'mismatch'
        locations = [{'stage':stage, 'line':exc.line}]
        for key in ['allocation_line', 'conflicting_line']:
            if exc.details.get(key):
                locations.append({'stage':stage,'line':exc.details[key]})
        result['findings'].append(finding(exc.rule_id, status,
            'bounded_symbolic' if symbolic else 'bounded_execution', str(exc),
            stage=stage, locations=locations, details={'case':case.name,**exc.details},
            scope='当前形状的符号路径观察，需确认可达性。' if symbolic else '当前用例在已支持的逻辑执行模型中触发。'))

    def coverage_finding(stage, case, pos, record, method):
        if record['status'] != 'mismatch':
            return
        result['findings'].append(finding('MEM-OUTPUT-UNWRITTEN', 'mismatch', method,
            f"输出参数 {pos} 有 {record['unwritten']} 个元素未写入", stage=stage,
            locations=[{'stage':stage,'line':line} for line in record['writer_lines']] or [{'stage':stage,'line':None}],
            details={'case':case.name,'output':pos,**record},
            scope='当前形状、纯输出契约与实际执行的控制路径。'))
    engines={}
    for stage in selected:
        try:
            if stage in STAGES[:3] and request.get('tir_backend') == 'tvm':
                from .tvm_interpreter import TVMInterpreter
                if stage not in official_functions:
                    raise Unsupported('官方 TIR 载入/检查未通过，见 tir_verification；未回退 AST')
                engines[stage] = TVMInterpreter(official_functions[stage],max_steps=request['max_steps'])
            elif stage in STAGES[3:] and request.get('mlir_backend') == 'official':
                from .mlir_object_interpreter import MLIRObjectInterpreter
                if stage not in official_mlir:
                    raise Unsupported('官方 MLIR 读取/验证未通过，见 mlir_python_verification；未回退文本执行')
                engines[stage] = MLIRObjectInterpreter(official_mlir[stage],sources[stage],max_steps=request['max_steps'])
            else:
                engines[stage]=(TIRInterpreter(sources[stage],max_steps=request['max_steps'],metadata=group.metadata.get('tir_metadata',{}).get(stage))
                                if stage in STAGES[:3] else MLIRInterpreter(sources[stage]))
            result['stage_support'][stage]={'status':'supported','backend':
                'tvm_tir_nodes' if stage in official_functions else
                'mlir_operation_objects' if stage in official_mlir else ('tir_ast' if stage in STAGES[:3] else 'mlir_text')}
        except Unsupported as exc:
            result['stage_support'][stage]={'status':'unsupported','reason':str(exc)}
            unsupported(stage, exc)
    arrays={}
    def save_array(name,value):
        result.setdefault('array_dtypes',{})[name]=str(value.dtype)
        arrays[name]=(np.array(value,copy=True).view(np.uint16 if value.dtype.itemsize==2 else np.uint8)
                      if value.dtype.name in {'bfloat16','float8_e4m3fn'} else np.array(value,copy=True))
    for i,case in enumerate(cases):
        for j,a in enumerate(case.arguments):
            if j not in case.outputs or j in case.inouts:
                save_array(f'case{i}_arg{j}',a)
        for pos,a in zip(case.outputs,case.expected):
            save_array(f'case{i}_expected{pos}',a)
    observed_shapes=set()
    any_mismatch=False
    symbolic_outputs={}
    # Symbolically enumerate each distinct shape, preserving memory provenance.
    for case in cases:
        shape_key=str([(a.shape,a.dtype.str) for a in case.arguments])+str(case.symbols)+str(case.outputs)+str(case.inouts)
        if shape_key in observed_shapes:
            continue
        observed_shapes.add(shape_key)
        outputs={}
        for stage,engine in engines.items():
            try:
                buffers=make_buffers(case.arguments,case.outputs,True,Budget(request['max_steps']),case.inouts)
                info=engine.run(buffers,case.symbols,True)
                outputs[stage]=buffers
                symbolic_outputs[(shape_key, stage)]=buffers
                cov={str(pos):coverage(buffers[pos]) for pos in case.outputs}
                bad_inputs=[pos for pos,b in enumerate(buffers) if pos not in case.outputs and pos not in case.inouts and np.any(b.writes)]
                if bad_inputs:
                    raise SemanticFault(f'修改了只读输入参数 {bad_inputs}')
                for pos,record in cov.items():
                    result['symbolic'].append({'kind':'coverage','case':case.name,'stage':stage,
                                                'output':int(pos),**record,**info})
                    any_mismatch |= record['status']=='mismatch'
                    coverage_finding(stage, case, int(pos), record, 'bounded_symbolic')
            except Unsupported as exc:
                result['symbolic'].append({'case':case.name,'stage':stage,'status':'unsupported','reason':str(exc)})
                unsupported(stage, exc, 'bounded_symbolic', case.name)
            except SemanticFault as exc:
                # Symbolic scalar branches inspect both alternatives. A memory
                # problem on such a path is not yet a feasible concrete witness.
                result['symbolic'].append({'case':case.name,'stage':stage,'status':'uncertain',
                                            'reason':str(exc)+'；需数值执行确认路径可达，符号观察不单独作为反例。'})
                fault(stage, case, exc, True)
        for left,right in zip(STAGES,STAGES[1:]):
            if left not in outputs or right not in outputs:
                continue
            for pos in case.outputs:
                comparison=compare_symbolic(outputs[left][pos],outputs[right][pos])
                result['symbolic'].append({'kind':'expression','case':case.name,'from':left,'to':right,
                                            'output':pos,**comparison})
                if comparison['status']=='uncertain':
                    diff=comparison['first_difference']
                    def features(expr, op):
                        if not isinstance(expr,list) or len(expr)<2:
                            return []
                        found=[expr[:2]+expr[2:]] if expr[0]==op else []
                        return found + [f for c in expr[2:] for f in features(c,op)]
                    rule='FLOW-EXPRESSION-DIFFERENCE'
                    if features(diff['left'],'input') != features(diff['right'],'input'):
                        rule='FLOW-SOURCE-DIFFERENCE'
                    elif features(diff['left'],'cast') != features(diff['right'],'cast'):
                        rule='PRECISION-CONVERSION-DIFFERENCE'
                    locations=[{'stage':s,'line':diff[k+'_writer_line']} for s,k in [(left,'left'),(right,'right')]]
                    if rule=='PRECISION-CONVERSION-DIFFERENCE':
                        for s,k in [(left,'left'),(right,'right')]:
                            locations.extend({'stage':s,'line':t['line']} for t in diff[k+'_trace']
                                             if t['action'] in {'arith.truncf','arith.extf','T.Cast'})
                    result['findings'].append(finding(rule,'uncertain','bounded_symbolic',
                        '相邻阶段输出表达式不同；需数值反例或合法改写依据',
                        locations=locations,
                        details={'case':case.name,'output':pos,'from':left,'to':right,**diff},
                        scope='仅当前有限形状；加法重排、初始化表达形式等可能合法。'))
    if request['dynamic']:
        result['executed']=bool(engines)
        for i,case in enumerate(cases):
            shape_key=str([(a.shape,a.dtype.str) for a in case.arguments])+str(case.symbols)+str(case.outputs)+str(case.inouts)
            for stage,engine in engines.items():
                try:
                    buffers=make_buffers(case.arguments,case.outputs,False,Budget(request['max_steps']),case.inouts)
                    info=engine.run(buffers,case.symbols,False)
                    bad_inputs=[pos for pos,b in enumerate(buffers) if pos not in case.outputs and pos not in case.inouts and np.any(b.writes)]
                    if bad_inputs:
                        raise SemanticFault(f'修改了只读输入参数 {bad_inputs}')
                    for pos,expected in zip(case.outputs,case.expected):
                        buf=buffers[pos]
                        cov=coverage(buf)
                        record=compare(buf.values,expected,case.output_comparisons.get(pos,case.comparison))
                        status='match' if record['match'] and cov['status']=='match' else 'mismatch'
                        result['tests'].append({'case':case.name,'stage':stage,'output':pos,'status':status,
                                                'backend':('tvm_tir_nodes' if stage in official_functions else
                                                           'mlir_operation_objects' if stage in official_mlir else
                                                           'tir_interpreter' if stage in STAGES[:3] else 'mlir_interpreter'),
                                                'coverage':cov,**record,**info})
                        save_array(f'case{i}_{stage}_out{pos}',buf.values)
                        any_mismatch |= status=='mismatch'
                        coverage_finding(stage, case, pos, cov, 'bounded_execution')
                        if not record['match']:
                            diff=record.get('first_difference',{})
                            index=tuple(diff.get('index',[]))
                            sym=symbolic_outputs.get((shape_key,stage))
                            chain=provenance(sym[pos].expressions[index]) if sym is not None and 'index' in diff else []
                            writer=int(buf.last_lines[index]) if 'index' in diff else None
                            result['findings'].append(finding('NUM-OUTPUT-DIFFERENCE','mismatch','bounded_execution',
                                '实际输出与独立参考不一致', stage=stage,line=writer or None,
                                details={'case':case.name,'output':pos,**record,'source_chain':chain,
                                         ('mlir_location' if stage in official_mlir else 'tvm_location'):
                                             getattr(engine,'writer_locations',{}).get((pos,index)),
                                         'source_chain_limit':64,'arrays_member':'arrays.npz'},
                                scope='当前输入、参考实现及显式比较契约；不直接归因到某个编译 pass。'))
                except Unsupported as exc:
                    result['tests'].append({'case':case.name,'stage':stage,'status':'unsupported','reason':str(exc)})
                    unsupported(stage, exc, 'bounded_execution', case.name)
                except SemanticFault as exc:
                    result['tests'].append({'case':case.name,'stage':stage,'status':'mismatch','reason':str(exc)})
                    any_mismatch=True
                    fault(stage, case, exc, False)
    stream=io.BytesIO()
    np.savez_compressed(stream,**arrays)
    result['_evidence']['arrays.npz']=base64.b64encode(stream.getvalue()).decode()
    result['arrays_member']='arrays.npz'
    result['arrays_index']={name:{'shape':list(a.shape),'dtype':str(a.dtype),
                                   'sha256':hashlib.sha256(a.tobytes()).hexdigest()} for name,a in arrays.items()}
    unsupported=(any(v['status']=='unsupported' for v in result['stage_support'].values())
                 or any(v['status']=='unsupported' for v in result['tests']+result['symbolic']))
    result['status']='mismatch' if any_mismatch else ('unsupported' if unsupported else 'passed')
    result['complete']=not unsupported and not any(s['status']=='uncertain' for s in result['symbolic'])
    result['has_mismatch']=bool(any_mismatch)
    result['case_count']=len(cases)
    result['stage_sha256']={stage:hashlib.sha256(source.encode()).hexdigest() for stage,source in sources.items()}
    return result


def main():
    request_path,response_path=map(Path,sys.argv[1:3])
    request=json.loads(request_path.read_text())
    log=io.StringIO()
    try:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            if request.get('verify_only'):
                from .mlir_frontend import verify_only
                result=verify_only(request)
            elif request.get('translation'):
                from .translation import execute_translation
                result=execute_translation(request)
            else:
                result=execute(request)
    except ImportError as exc:
        result={'status':'unsupported','executed':False,'reason':f'执行依赖不可用：{exc}'}
    except Exception as exc:
        from .execution import Unsupported
        result={'status':'unsupported' if isinstance(exc,Unsupported) else 'error',
                'executed':False,'reason':f'{type(exc).__name__}: {exc}'}
        if not isinstance(exc,Unsupported):
            result['traceback']=traceback.format_exc()
    if log.getvalue():
        result['worker_log']=log.getvalue()[-20000:]
    response_path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
