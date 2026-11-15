import numpy as np
from tir_mlir_checker.cases import TestCase, Comparison
from tir_mlir_checker.execution import dtype
compile_options = {"target":"riscv", "out_idx":[1]}
POLICY=Comparison("tolerance",rtol=1e-4,atol=1e-5,signed_zero_equal=True)
def tc(name,args,outputs,expected,policy=POLICY):
    return TestCase(name,tuple(args),tuple(outputs),tuple(expected),comparison=policy)

def make_cases(seed):
    rng=np.random.default_rng(seed)
    for mode in ('random','positive','ramp'):
        x=(rng.normal(size=(2,33)).astype('float32') if mode=='random' else np.ones((2,33),dtype='float32') if mode=='positive' else np.arange(66,dtype='float32').reshape(2,33)-10)
        y=x.sum(axis=1,dtype='float32')
        yield tc(mode,(x,np.empty_like(y)),(1,),(y,))
