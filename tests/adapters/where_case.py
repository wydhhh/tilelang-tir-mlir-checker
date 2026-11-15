"""Where is a test adapter, not part of the common execution/checking engine."""
import numpy as np

from tir_mlir_checker.cases import Comparison, TestCase


def make_cases(seed):
    rng=np.random.default_rng(seed)
    size=2048
    x=(np.arange(size,dtype=np.float32)+np.float32(0.25)).copy()
    y=(-np.arange(size,dtype=np.float32)-np.float32(4096.5)).copy()
    patterns={
        "zero":np.zeros(size,dtype=np.uint8),
        "nonzero":np.ones(size,dtype=np.uint8),
        "alternating":(np.arange(size)%2).astype(np.uint8),
        "all_byte_classes":np.resize(np.array([0,1,2,127,128,255],dtype=np.uint8),size),
        "random":rng.integers(0,256,size,dtype=np.uint8),
        "block_boundaries":np.zeros(size,dtype=np.uint8),
    }
    patterns["block_boundaries"][[0,511,512,1023,1024,1535,1536,2047]]=255
    for label,cond in patterns.items():
        yield TestCase(label,(cond.copy(),x.copy(),y.copy(),np.empty(size,dtype=np.float32)),
                       (3,),(np.where(cond!=0,x,y),))
    # Raw f32 bit patterns exercise NaN payloads, +/-Inf and +/-0. Where only
    # selects/copies values, so expected output is compared bit-for-bit.
    xbits=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x3f800000,0xbf800000],dtype=np.uint32)
    xs=np.resize(xbits.view(np.float32),size).copy()
    ys=np.resize(np.roll(xbits,3).view(np.float32),size).copy()
    for label,cond in [("special_zero",patterns["zero"]),("special_nonzero",patterns["nonzero"]),
                       ("special_mixed",patterns["all_byte_classes"])]:
        yield TestCase(label,(cond.copy(),xs.copy(),ys.copy(),np.empty(size,dtype=np.float32)),
                       (3,),(np.where(cond!=0,xs,ys),),comparison=Comparison(mode="bitwise"))
