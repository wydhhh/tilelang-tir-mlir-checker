"""Minimal numeric/initialization contract, independent oracles and counterexamples."""
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tir_mlir_checker import check
from tir_mlir_checker.execution import (Budget,Expr,apply,cast,dtype,make_buffers,mlir_cast,mlir_cmpi)
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter
from tests.support.calibration import models,simulate,sum_ir,f32_to_bf16_bits,f32_to_f16_bits
from test_execution import write_group

HAS_TVM=importlib.util.find_spec('tilelang') is not None
HAS_MLIR=importlib.util.find_spec('mlir') is not None


def case_file(path,cases):
    def array(a):
        return 'np.frombuffer(bytes.fromhex('+repr(a.tobytes().hex())+'),dtype='+repr(a.dtype.name)+').reshape('+repr(a.shape)+').copy()'
    lines=['import numpy as np','from tir_mlir_checker.cases import TestCase,Comparison','def make_cases(seed):']
    for c in cases:
        arguments='('+','.join(array(a) for a in c.arguments)+',)'
        expected='('+','.join(array(a) for a in c.expected)+',)'
        lines.append('    yield TestCase('+repr(c.name)+','+arguments+','+repr(c.outputs)+','+expected+',comparison=Comparison(**'+repr(asdict(c.comparison))+'))')
    path.write_text('\n'.join(lines)+'\n');return path


class NumericContractTests(unittest.TestCase):
    def test_integer_sign_zero_extension_and_i1(self):
        for v in (0,1,127,128,255):
            self.assertEqual(int(mlir_cast('extui',v,'i8','i64')),v)
            self.assertEqual(int(mlir_cast('extsi',v,'i8','i64')),v if v<128 else v-256)
        self.assertEqual(int(mlir_cast('extsi',True,'i1','i32')),-1)
        self.assertEqual(int(mlir_cast('extui',True,'i1','i32')),1)
        self.assertEqual(int(mlir_cast('trunci',3,'i64','i1')),1)
        self.assertEqual(int(mlir_cast('trunci',2,'i64','i1')),0)
        self.assertEqual(int(mlir_cast('trunci',2,'i64','bool')),0)

    def test_signed_unsigned_comparison_and_float_conversion(self):
        self.assertTrue(mlir_cmpi('slt',255,1,'i8'))
        self.assertFalse(mlir_cmpi('ult',255,1,'i8'))
        self.assertTrue(mlir_cmpi('slt',True,False,'i1'))
        self.assertTrue(mlir_cmpi('ugt',True,False,'i1'))
        self.assertEqual(mlir_cast('sitofp',255,'i8','f32'),-1)
        self.assertEqual(mlir_cast('uitofp',255,'i8','f32'),255)
        self.assertTrue(mlir_cmpi('ult',2**63-1,-2**63,'i64'))
        self.assertFalse(mlir_cmpi('eq',2**63-1,-2**63,'i64'))

    def test_exact_wrap_and_int64_division(self):
        for name,width in [('i8',8),('i32',32),('i64',64)]:
            self.assertEqual(int(apply('add',[cast(2**(width-1)-1,name),cast(1,name)],name)),-2**(width-1))
            self.assertEqual(int(cast(2**width+3,name)),3)
        for a,b in [(2**63-1,3),(-2**63+1,7),(-2**63+1,-7)]:
            self.assertEqual(int(apply('floordiv',[np.int64(a),np.int64(b)],'i64')),a//b)
            q=(abs(a)//abs(b))*(-1 if (a<0)!=(b<0) else 1)
            self.assertEqual(int(apply('rem',[np.int64(a),np.int64(b)],'i64')),a-q*b)
        self.assertEqual(int(cast(2**64-1,'uint64')),2**64-1)

    def test_symbolic_signedness_cannot_collapse(self):
        x=Expr('input',(0,(0,)),'i8')
        self.assertNotEqual(mlir_cast('extsi',x,'i8','i64'),mlir_cast('extui',x,'i8','i64'))
        self.assertNotEqual(mlir_cmpi('slt',x,np.int8(1),'i8'),mlir_cmpi('ult',x,np.int8(1),'i8'))

    def test_low_precision_oracles_ties_and_specials(self):
        self.assertEqual(f32_to_bf16_bits(0x3f808000),0x3f80)
        self.assertEqual(f32_to_bf16_bits(0x3f818000),0x3f82)
        self.assertEqual(f32_to_f16_bits(0x3f801000),0x3c00)
        self.assertEqual(f32_to_f16_bits(0x3f803000),0x3c02)
        for m in models()[:2]:
            engine=MLIRInterpreter(m.mlir)
            for case in m.cases:
                buffers=make_buffers(case.arguments,case.outputs,False,Budget())
                engine.run(buffers)
                from tir_mlir_checker.cases import compare
                self.assertTrue(compare(buffers[1].values,case.expected[0],case.comparison)['match'])

    def test_new_integer_operations_in_lightweight_text_backend(self):
        for m in (m for m in models()[2:] if not m.name.startswith('sum_')):
            with self.subTest(model=m.name):
                engine=MLIRInterpreter(m.mlir)
                for case in m.cases:
                    buffers=make_buffers(case.arguments,case.outputs,False,Budget());engine.run(buffers)
                    np.testing.assert_array_equal(buffers[1].values,case.expected[0])

    def test_capability_inventory_matches_selected_backend(self):
        from tir_mlir_checker.rules import tir_inventory,mlir_inventory
        from tir_mlir_checker.analysis import mlir_facts
        from tir_mlir_checker.trace_io import STAGES
        tir,mlir=sum_ir(block=True)
        _,gaps=tir_inventory(tir,STAGES[0])
        self.assertTrue(any(g['rule_id']=='CAP-TIR-CALL' for g in gaps))
        self.assertFalse(tir_inventory(tir,STAGES[0],backend='tvm')[1])
        facts=mlir_facts(mlir)
        self.assertTrue(mlir_inventory(mlir,STAGES[3],facts)[1])
        self.assertFalse(mlir_inventory(mlir,STAGES[3],facts,backend='official')[1])


@unittest.skipUnless(HAS_TVM,'requires matching official TVM')
class OfficialNumericTIRTests(unittest.TestCase):
    def test_all_numeric_and_reduction_models(self):
        for model in models():
            with self.subTest(model=model.name):
                rows,_=simulate(model,'tvm')
                self.assertTrue(all(r['match'] for r in rows),rows)

    def test_scheduled_reduction_init_does_not_silently_run(self):
        from tests.support.calibration import Model
        m=next(m for m in models() if m.name=='sum_block_init')
        # Same complete domain but indirect binding deliberately stays outside MVP.
        changed=m.tir.replace('T.axis.reduce(4,k)','T.axis.reduce(4,3-k)')
        from tir_mlir_checker.execution import Unsupported
        with self.assertRaisesRegex(Unsupported,'归约轴映射'):
            simulate(Model('scheduled',changed,m.mlir,m.cases),'tvm')

    @unittest.skipUnless(HAS_MLIR,'requires both official frontends')
    def test_standard_block_init_full_five_stage_check(self):
        m=next(m for m in models() if m.name=='sum_block_init')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);trace=write_group(root/'trace',[m.tir]*3+[m.mlir]*2)
            r=check(trace,case=case_file(root/'case.py',m.cases),dynamic=True,
                    tir_backend='tvm',mlir_backend='official',timeout=120)
            self.assertEqual(r['tool_status'],'ok',r['dynamic'])
            self.assertEqual(r['dynamic']['status'],'passed')
            self.assertFalse(any(f['status']=='unsupported' for f in r['findings']))


@unittest.skipUnless(HAS_MLIR,'requires matching official MLIR Python')
class OfficialNumericMLIRTests(unittest.TestCase):
    def test_all_numeric_and_reduction_models(self):
        for model in models():
            with self.subTest(model=model.name):
                rows,_=simulate(model,'mlir')
                self.assertTrue(all(r['match'] for r in rows),rows)

    def test_reduction_mutations_have_diagnostics_and_counterexamples(self):
        m=next(m for m in models() if m.name=='sum_axis1')
        variants={
            'wrong_axis':sum_ir(axis=0)[1],
            'wrong_initial':sum_ir(initial=1.)[1],
            'early_precision':sum_ir(low_precision=True)[1],
            'wrong_order':sum_ir(reverse=True)[1],
            'missing_init':sum_ir(carried=False,missing_init=True)[1],
        }
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);adapter=case_file(root/'case.py',m.cases)
            for name,source in variants.items():
                with self.subTest(name=name):
                    trace=write_group(root/name,[m.tir]*3+[source,m.mlir])
                    r=check(trace,case=adapter,dynamic=True,mlir_backend='official',stages=['03_mlir_linalg'],timeout=120)
                    self.assertEqual(r['mlir_python_verification']['status'],'passed')
                    self.assertEqual(r['tool_status'],'mismatch',r['dynamic'])
                    self.assertEqual(r['semantic_conclusion'],'bounded_counterexample')
                    self.assertTrue(any(f['rule_id'] in {'NUM-OUTPUT-DIFFERENCE','MEM-UNINITIALIZED'} for f in r['findings']))

    def test_carried_sum_is_not_a_known_capability_gap(self):
        m=next(m for m in models() if m.name=='sum_axis1')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            trace=write_group(root/'trace',[m.tir]*3+[m.mlir]*2)
            r=check(trace,case=case_file(root/'case.py',m.cases),dynamic=True,
                    mlir_backend='official',stages=['03_mlir_linalg'],timeout=120)
            self.assertEqual(r['dynamic']['status'],'passed')
            self.assertFalse(any(f['rule_id']=='CAP-LOOP-ARGS' for f in r['findings']))


if __name__=='__main__':unittest.main()
