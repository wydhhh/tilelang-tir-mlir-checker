import numpy as np
from tir_mlir_checker.cases import TestCase, Comparison
from tir_mlir_checker.execution import dtype
compile_options = {"target":"riscv", "out_idx":[3]}
POLICY=Comparison("tolerance",rtol=1e-4,atol=1e-5,signed_zero_equal=True)
def tc(name,args,outputs,expected,policy=POLICY):
    return TestCase(name,tuple(args),tuple(outputs),tuple(expected),comparison=policy)

def make_cases(seed):
    rng=np.random.default_rng(seed)
    for mode in ('all0','all1','mixed'):
        c=(np.zeros(35,dtype='uint8') if mode=='all0' else np.ones(35,dtype='uint8') if mode=='all1' else rng.integers(0,256,size=35,dtype='uint8'))
        x=np.arange(35,dtype='float32')+1; y=-x-2
        z=np.where(c!=0,x,y)
        yield tc(mode,(c,x,y,np.empty_like(z)),(3,),(z,),Comparison('bitwise'))
