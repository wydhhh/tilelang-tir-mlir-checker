"""Independent NumPy references for the saved [2,256] Sum trace.

This is a test adapter, never used to dispatch execution semantics.
Tolerance is explicit because reduction trees may legitimately differ.
"""
import numpy as np

from tir_mlir_checker.cases import Comparison, TestCase


def make_cases(seed):
    rng = np.random.default_rng(seed)
    rows = np.arange(512, dtype=np.float32).reshape(2, 256)
    cases = {
        'zeros': np.zeros((2, 256), np.float32),
        'distinct_rows': rows - np.float32(200),
        'sparse_edges': np.zeros((2, 256), np.float32),
        'alternating': (np.where(np.arange(512) % 2, -1, 1) * np.arange(512)).astype(np.float32).reshape(2, 256),
        'random': rng.normal(size=(2, 256)).astype(np.float32),
    }
    cases['sparse_edges'][0, [0, 255]] = [3, 17]
    cases['sparse_edges'][1, [0, 127, 255]] = [-13, 7, 29]
    for label, x in cases.items():
        yield TestCase(label, (x, np.empty(2, np.float32)), (1,),
            (np.sum(x, axis=1, dtype=np.float32),),
            comparison=Comparison(mode='tolerance', rtol=1e-4, atol=1e-4,
                                  equal_nan=False, signed_zero_equal=True))
