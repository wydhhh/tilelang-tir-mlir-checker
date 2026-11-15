"""Small numeric/reduction calibration corpus; not shipped in the runtime wheel.

Run in the matching TVM/MLIR environment. --native additionally uses the existing
RISC-V toolchain. References here never call checker execution helpers.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import tempfile
import numpy as np
from tir_mlir_checker.cases import Comparison, TestCase, compare

EXACT_NAN = Comparison('tolerance', equal_nan=True)  # zero tolerance, signed zero checked


def f32_to_bf16_bits(raw):
    """Independent integer RNE oracle; NaN payload is outside arithmetic contract."""
    raw = int(raw)
    if raw & 0x7fffffff > 0x7f800000: return (raw >> 16 & 0x8000) | 0x7fc0
    return ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16) & 0xffff


def f32_to_f16_bits(raw):
    value = struct.unpack('<f', struct.pack('<I', int(raw)))[0]
    if math.isnan(value): return 0x7e00 | (int(raw) >> 16 & 0x8000)
    try: return struct.unpack('<H', struct.pack('<e', value))[0]
    except OverflowError: return 0xfc00 if value < 0 else 0x7c00


def promote_bits(raw, kind):
    if kind == 'bf16': return int(raw) << 16
    return struct.unpack('<I',struct.pack('<f',struct.unpack('<e',struct.pack('<H',int(raw)))[0]))[0]


def sequential_sum(x, axis, initial=0.0, *, reverse=False):
    """Independent explicit f32 left fold, deliberately not np.sum's tree."""
    shape = x.shape[:axis] + x.shape[axis+1:]
    answer = np.full(shape, initial, dtype='float32')
    for index in np.ndindex(shape):
        value = np.float32(initial)
        for k in (reversed(range(x.shape[axis])) if reverse else range(x.shape[axis])):
            item = x[index[:axis]+(k,)+index[axis:]]
            value = np.float32(float(value)+float(item))
        answer[index] = value
    return answer


def elementwise_ir(input_shape, source, target, body, tir_expression):
    n = input_shape[0]
    mlir = f'''module {{
  func.func @main(%x: memref<{n}x{source}>, %y: memref<{n}x{target}>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cn = arith.constant {n} : index
    scf.for %i = %c0 to %cn step %c1 {{
      %xv = memref.load %x[%i] : memref<{n}x{source}>
{body}
      memref.store %v, %y[%i] : memref<{n}x{target}>
    }}
    return
  }}
}}
'''
    types={'f32':'float32','bf16':'bfloat16','f16':'float16','i8':'int8','i64':'int64','i32':'int32'}
    tir=f'''@I.ir_module
class Module:
    @T.prim_func
    def main(x: T.Buffer(({n},), "{types[source]}"), y: T.Buffer(({n},), "{types[target]}")):
        for i in T.serial({n}):
            y[i] = {tir_expression}
'''
    return tir, mlir


def sum_ir(axis=1, initial=0.0, *, carried=True, low_precision=False, reverse=False, block=False, missing_init=False):
    # Square input makes a swapped axis structurally legal but numerically wrong.
    i,j=('%i','%k') if axis==1 else ('%k','%i')
    index='%rk' if reverse else '%k'
    i,j=(i.replace('%k',index),j.replace('%k',index))
    prepare='      %c3 = arith.constant 3 : index\n      %rk = arith.subi %c3, %k : index\n' if reverse else ''
    precision=('      %low = arith.truncf %next : f32 to bf16\n      %up = arith.extf %low : bf16 to f32\n' if low_precision else '')
    final='%up' if low_precision else '%next'
    if carried:
        loop=f'''    scf.for %i = %c0 to %c4 step %c1 {{
      %sum = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %initial) -> (f32) {{
{prepare}      %item = memref.load %x[{i}, {j}] : memref<4x4xf32>
      %next = arith.addf %acc, %item : f32
{precision}      scf.yield {final} : f32
      }}
      memref.store %sum, %y[%i] : memref<4xf32>
    }}'''
    else:
        init='' if missing_init else '      memref.store %initial, %y[%i] : memref<4xf32>\n'
        loop=f'''    scf.for %i = %c0 to %c4 step %c1 {{
{init}      scf.for %k = %c0 to %c4 step %c1 {{
{prepare}      %acc = memref.load %y[%i] : memref<4xf32>
      %item = memref.load %x[{i}, {j}] : memref<4x4xf32>
      %next = arith.addf %acc, %item : f32
{precision}      memref.store {final}, %y[%i] : memref<4xf32>
      }}
    }}'''
    mlir=f'''module {{
  func.func @main(%x: memref<4x4xf32>, %y: memref<4xf32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %initial = arith.constant {initial:.8e} : f32
{loop}
    return
  }}
}}
'''
    ii,jj=('i','k') if axis==1 else ('k','i')
    if reverse: ii,jj=ii.replace('k','3-k'),jj.replace('k','3-k')
    expression=f'y[i] + x[{ii}, {jj}]'
    if low_precision: expression=f'T.Cast("float32", T.Cast("bfloat16", {expression}))'
    if block:
        # Unscheduled TIR reduction block; initialize once per spatial point.
        ii,jj=('vi','vk') if axis==1 else ('vk','vi')
        tir=f'''@I.ir_module
class Module:
    @T.prim_func
    def main(x: T.Buffer((4,4), "float32"), y: T.Buffer((4,), "float32")):
        for i, k in T.grid(4,4):
            with T.block("sum"):
                vi = T.axis.spatial(4,i)
                vk = T.axis.reduce(4,k)
                with T.init():
                    y[vi] = T.float32({initial})
                y[vi] = y[vi] + x[{ii},{jj}]
'''
    else:
        init='' if missing_init else f'            y[i] = T.float32({initial})\n'
        tir=f'''@I.ir_module
class Module:
    @T.prim_func
    def main(x: T.Buffer((4,4), "float32"), y: T.Buffer((4,), "float32")):
        for i in T.serial(4):
{init}            for k in T.serial(4):
                y[i] = {expression}
'''
    return tir,mlir


@dataclass
class Model:
    name:str
    tir:str
    mlir:str
    cases:list


def models():
    result=[]
    rng=np.random.default_rng(71)
    # Halfway ties, subnormals, overflow, signed zero, infinities and NaNs.
    raw=np.array([0,0x80000000,0x3f800000,0x3f801000,0x3f803000,0x3f808000,0x3f818000,
        0x33800000,0x33000000,0x387fc000,0x38800000,0x477fe000,0x477ff000,
        0x00008000,0x00018000,0x7f7fffff,0x7f800000,0xff800000,0x7fc12345,0xffa12345],dtype='uint32')
    raw=np.concatenate((raw,rng.integers(0,2**32,108,dtype='uint32')))
    x=raw.view('float32')
    for kind,oracle in [('bf16',f32_to_bf16_bits),('f16',f32_to_f16_bits)]:
        bits=np.array([oracle(v) for v in raw],dtype='uint16')
        expected=np.array([promote_bits(v,kind) for v in bits],dtype='uint32').view('float32')
        tir,mlir=elementwise_ir(x.shape,'f32','f32',f'      %low = arith.truncf %xv : f32 to {kind}\n      %v = arith.extf %low : {kind} to f32',
            f'T.Cast("float32", T.Cast("{"bfloat16" if kind=="bf16" else "float16"}", x[i]))')
        case=TestCase(kind+'_ties_edges_random',(x.copy(),np.empty_like(x)),(1,),(expected,),comparison=EXACT_NAN)
        result.append(Model('roundtrip_'+kind,tir,mlir,[case]))
    raw_i=np.array([0,1,127,128,255,2,129,254],dtype='uint8').view('int8')
    for opname,target,expected,expr in [
        ('extsi','i64',raw_i.astype('int64'),'T.Cast("int64", x[i])'),
        ('extui','i64',raw_i.view('uint8').astype('int64'),'T.Cast("int64", T.Cast("uint8", x[i]))'),
        ('sitofp','f32',raw_i.astype('float32'),'T.Cast("float32", x[i])'),
        ('uitofp','f32',raw_i.view('uint8').astype('float32'),'T.Cast("float32", T.Cast("uint8", x[i]))'),
    ]:
        tir,mlir=elementwise_ir(raw_i.shape,'i8',target,f'      %v = arith.{opname} %xv : i8 to {target}',expr)
        result.append(Model(opname,tir,mlir,[TestCase(opname,(raw_i.copy(),np.empty_like(expected)),(1,),(expected,))]))
    for pred in ('slt','ult'):
        expected=np.array([int(int(v)<1) for v in (raw_i if pred=='slt' else raw_i.view('uint8'))],dtype='int64')
        expr='T.Cast("int64", '+('x[i]' if pred=='slt' else 'T.Cast("uint8", x[i])')+' < '+('T.int8(1)' if pred=='slt' else 'T.uint8(1)')+')'
        body=f'''      %one = arith.constant 1 : i8
      %test = arith.cmpi {pred}, %xv, %one : i8
      %yes = arith.constant 1 : i64
      %no = arith.constant 0 : i64
      %v = arith.select %test, %yes, %no : i64'''
        tir,mlir=elementwise_ir(raw_i.shape,'i8','i64',body,expr)
        result.append(Model('cmp_'+pred,tir,mlir,[TestCase(pred,(raw_i.copy(),np.empty_like(expected)),(1,),(expected,))]))
    wide=np.array([-129,-128,-1,0,127,128,255,256],dtype='int64')
    expected=np.array([((int(v)+128)%256)-128 for v in wide],dtype='int8')
    tir,mlir=elementwise_ir(wide.shape,'i64','i8','      %v = arith.trunci %xv : i64 to i8','T.Cast("int8", x[i])')
    result.append(Model('trunci',tir,mlir,[TestCase('low_bits',(wide,np.empty_like(expected)),(1,),(expected,))]))
    xi=np.array([2**31-1,-2**31,-1,0],dtype='int32');expected=np.array([-2**31,-2**31+1,0,1],dtype='int32')
    tir,mlir=elementwise_ir(xi.shape,'i32','i32','      %one = arith.constant 1 : i32\n      %v = arith.addi %xv, %one : i32','x[i]+T.int32(1)')
    result.append(Model('integer_wrap',tir,mlir,[TestCase('i32_wrap',(xi,np.empty_like(xi)),(1,),(expected,))]))
    wide=np.array([-1,-2**63,-2**63+1,2**63-1,0,1,2,3],dtype='int64')
    expected=np.array([int(v)%2 for v in wide],dtype='int64')
    tir,mlir=elementwise_ir(wide.shape,'i64','i64',
        '      %low = arith.trunci %xv : i64 to i1\n      %v = arith.extui %low : i1 to i64',
        'x[i] % T.int64(2)')
    result.append(Model('trunci_i1',tir,mlir,[TestCase('lowest_bit',(wide.copy(),np.empty_like(wide)),(1,),(expected,))]))
    expected=np.array([int((int(v)%(2**64))<2**63) for v in wide],dtype='int64')
    tir,mlir=elementwise_ir(wide.shape,'i64','i64',
        '''      %limit = arith.constant -9223372036854775808 : i64
      %test = arith.cmpi ult, %xv, %limit : i64
      %v = arith.extui %test : i1 to i64''',
        'T.Cast("int64", T.Cast("uint64", x[i]) < T.Cast("uint64", T.int64(-9223372036854775808)))')
    result.append(Model('cmp_u64_boundary',tir,mlir,[TestCase('u64_high_bit',(wide.copy(),np.empty_like(wide)),(1,),(expected,))]))
    # Different rows/columns, cancellation and ties make axis/order/precision bugs observable.
    data=[('distinct',np.arange(16,dtype='float32').reshape(4,4)+np.float32(.12345)),
          ('cancellation',np.array([[2**24,1,-2**24,1],[1,.00390625,.00390625,-1],[-0.,0.,-0.,0.],[1e-20,1,-1,1e-20]],dtype='float32')),
          ('special',np.array([[np.inf,-np.inf,1,2],[-np.inf,0,1,2],[np.nan,1,2,3],[-0.,-0.,-0.,-0.]],dtype='float32'))]
    for name,axis,initial,block in [('sum_axis1',1,0.,False),('sum_axis0',0,0.,False),
                                  ('sum_nonzero_init',1,3.25,False),('sum_block_init',1,0.,True)]:
        tir,mlir=sum_ir(axis,initial,block=block)
        cases=[]
        for label,a in data:
            with np.errstate(all='ignore'):expected=sequential_sum(a,axis,initial)
            cases.append(TestCase(label,(a.copy(),np.empty_like(expected)),(1,),(expected,),comparison=EXACT_NAN))
        result.append(Model(name,tir,mlir,cases))
    return result


def simulate(model, backend):
    from tir_mlir_checker.execution import Budget, make_buffers
    if backend=='tvm':
        from tilelang import tvm
        from tvm.script import ir as I,tir as T
        from tir_mlir_checker.tvm_interpreter import TVMInterpreter
        obj=tvm.script.from_source(model.tir,extra_vars={'I':I,'T':T})
        fun=list(obj.functions.values())[0]
        assert tvm.tir.analysis.verify_ssa(fun) and tvm.tir.analysis.verify_well_formed(fun)
        engine=TVMInterpreter(fun)
    else:
        import mlir.ir as ir
        from tir_mlir_checker.mlir_frontend import DIALECTS
        import importlib
        for name in DIALECTS:importlib.import_module('mlir.dialects.'+name)
        ctx=ir.Context()
        with ctx:
            ctx.allow_unregistered_dialects=False;ctx.load_all_available_dialects()
            module=ir.Module.parse(model.mlir);assert module.operation.verify()
        from tir_mlir_checker.mlir_object_interpreter import MLIRObjectInterpreter
        engine=MLIRObjectInterpreter((ctx,module),model.mlir)
    rows=[];arrays={}
    for i,case in enumerate(model.cases):
        buffers=make_buffers(case.arguments,case.outputs,False,Budget())
        info=engine.run(buffers)
        for pos,expected in zip(case.outputs,case.expected):
            actual=buffers[pos].values
            record=compare(actual,expected,case.comparison)
            rows.append({'case':case.name,'output':pos,**record,'backend':backend,'execution':info})
            arrays[f'case{i}_out{pos}']=actual.copy()
    return rows,arrays


def calibrate(out, *, native=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records=[];arrays={}
    from tir_mlir_checker import __version__
    for model in models():
        print(model.name,flush=True)
        folder=out/model.name;folder.mkdir()
        (folder/'input.tir').write_text(model.tir);(folder/'input.mlir').write_text(model.mlir)
        item={'name':model.name,'models':{},'tir_sha256':hashlib.sha256(model.tir.encode()).hexdigest(),
              'mlir_sha256':hashlib.sha256(model.mlir.encode()).hexdigest()}
        for backend in ('tvm','mlir'):
            rows,values=simulate(model,backend)
            item['models'][backend]=rows
            arrays.update({model.name+'_'+backend+'_'+k:v for k,v in values.items()})
        if native:
            from tests.support.native import NativeMLIR
            with tempfile.TemporaryDirectory(prefix='numeric-native-') as temp:
                engine=NativeMLIR(model.mlir,'main',temp)
                item['native_info']=engine.info
                try:
                    rows=[]
                    for i,case in enumerate(model.cases):
                        for pos,actual,expected in zip(case.outputs,engine.run(case),case.expected):
                            rows.append({'case':case.name,'output':pos,**compare(actual,expected,case.comparison),
                                'vs_simulator':compare(actual,arrays[model.name+'_mlir_'+f'case{i}_out{pos}'],case.comparison)})
                            arrays[model.name+'_native_'+f'case{i}_out{pos}']=actual.copy()
                    item['models']['native']=rows
                finally:engine.close()
        records.append(item)
        (out/'result.json').write_text(json.dumps({'tool_version':__version__,'native_requested':native,
            'machine':platform.machine(),'models':records},ensure_ascii=False,indent=2))
    np.savez_compressed(out/'arrays.npz',**arrays)
    assert all(r['match'] and r.get('vs_simulator',{'match':True})['match'] for m in records for rows in m['models'].values() for r in rows)
    print('calibration passed',len(records),'models',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--native',action='store_true')
    args=p.parse_args();calibrate(args.out,native=args.native)
