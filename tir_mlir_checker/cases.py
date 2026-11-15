"""Public test-adapter protocol. Operator-specific code belongs in adapters."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .execution import Unsupported


@dataclass(frozen=True)
class Comparison:
    mode: str = "bitwise"  # bitwise | tolerance
    rtol: float = 0.0
    atol: float = 0.0
    equal_nan: bool = False
    signed_zero_equal: bool = False


@dataclass
class TestCase:
    name: str
    arguments: tuple[np.ndarray, ...]
    outputs: tuple[int, ...]
    expected: tuple[np.ndarray, ...]
    symbols: dict[str, int] = field(default_factory=dict)
    comparison: Comparison = field(default_factory=Comparison)
    output_comparisons: dict[int, Comparison] = field(default_factory=dict)
    inouts: tuple[int, ...] = ()  # initialized writable args; may include auxiliary workspace


def validate_case(case):
    if not isinstance(case, TestCase) or not isinstance(case.name, str) or not case.name:
        raise Unsupported("make_cases() 必须返回具有唯一名称的 TestCase")
    if (not case.arguments or not case.outputs or len(set(case.outputs)) != len(case.outputs)
            or len(case.expected) != len(case.outputs)
            or any(type(i) is not int or not 0 <= i < len(case.arguments) for i in case.outputs)):
        raise Unsupported("TestCase 的参数、输出位置或参考输出数量错误")
    if len(set(case.inouts))!=len(case.inouts) or any(type(i) is not int or not 0<=i<len(case.arguments) for i in case.inouts):
        raise Unsupported('inouts 必须为合法参数位置，保留捕获的初始状态')
    if not isinstance(case.comparison, Comparison) or case.comparison.mode not in {"bitwise","tolerance"}:
        raise Unsupported("比较模式必须为 bitwise 或 tolerance")
    if not np.isfinite(case.comparison.rtol) or not np.isfinite(case.comparison.atol) or min(case.comparison.rtol,case.comparison.atol) < 0:
        raise Unsupported("容差必须为有限非负值")
    for pos,policy in case.output_comparisons.items():
        if pos not in case.outputs or not isinstance(policy,Comparison) or policy.mode not in {'bitwise','tolerance'} or not np.isfinite(policy.rtol) or not np.isfinite(policy.atol) or min(policy.rtol,policy.atol)<0:
            raise Unsupported('output_comparisons 的输出位置或策略非法')
    for array in case.arguments:
        if not isinstance(array,np.ndarray) or (array.dtype.kind not in "biuf" and array.dtype.name not in {"bfloat16", "float8_e4m3fn"}) or not array.flags.c_contiguous or array.size > 1_000_000:
            raise Unsupported("参数必须为非 object 的连续数组且元素不超过 1000000")
    for position,expected in zip(case.outputs,case.expected):
        actual=case.arguments[position]
        if not isinstance(expected,np.ndarray) or expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise Unsupported("参考输出与输出参数的 shape/dtype 不同")
    if any(type(value) is not int for value in case.symbols.values()):
        raise Unsupported("动态符号必须显式绑定为整数")


def compare(actual,expected,policy):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"match":False,"reason":"shape/dtype mismatch"}
    bits_a=np.ascontiguousarray(actual).view(np.uint8).reshape(actual.shape+(actual.dtype.itemsize,))
    bits_b=np.ascontiguousarray(expected).view(np.uint8).reshape(expected.shape+(expected.dtype.itemsize,))
    bit_equal=np.all(bits_a==bits_b,axis=-1)
    if policy.mode=="bitwise":
        equal=bit_equal
    else:
        # Do not allow inf-inf to count as a finite tolerance comparison.
        if actual.dtype.kind=="f" or actual.dtype.name in {"bfloat16", "float8_e4m3fn"}:
            a,b=actual.astype(np.float64),expected.astype(np.float64)
            with np.errstate(all="ignore"):
                equal=(np.isfinite(a)&np.isfinite(b)&(np.abs(a-b)<=policy.atol+policy.rtol*np.abs(b)))|(np.isinf(a)&(a==b))
            if policy.equal_nan:
                equal |= np.isnan(a)&np.isnan(b)
            if not policy.signed_zero_equal:
                equal &= ~((a==0)&(b==0)&(np.signbit(a)!=np.signbit(b)))
        else:
            equal=actual==expected
    bad=np.argwhere(~equal)
    finite=np.isfinite(actual)&np.isfinite(expected)
    error=float(np.max(np.abs(actual[finite].astype(np.float64)-expected[finite].astype(np.float64)))) if np.any(finite) else None
    result={"match":not bool(len(bad)),"elements":int(actual.size),"unequal_elements":int(len(bad)),
            "bitwise_equal_elements":int(np.count_nonzero(bit_equal)),"max_finite_abs_error":error,
            "policy":{"mode":policy.mode,"rtol":policy.rtol,"atol":policy.atol,
                      "equal_nan":policy.equal_nan,"signed_zero_equal":policy.signed_zero_equal}}
    if len(bad):
        index=tuple(int(i) for i in bad[0])
        def detail(array):
            value=array[index]
            return {"value":repr(value.item()),"bits":np.asarray(value).tobytes().hex()}
        result["first_difference"]={"index":list(index),"actual":detail(actual),"expected":detail(expected)}
    return result


def cases_fingerprint(cases):
    """Bind replay to inputs, reference values and comparison policies, not a seed alone."""
    import hashlib
    import json
    from dataclasses import asdict
    digest = hashlib.sha256()
    for case in cases:
        contract = {'name': case.name, 'outputs': case.outputs, 'inouts': case.inouts,
                    'symbols': case.symbols, 'comparison': asdict(case.comparison),
                    'output_comparisons': {k: asdict(v) for k, v in case.output_comparisons.items()}}
        digest.update(json.dumps(contract, sort_keys=True).encode())
        for pos, array in enumerate(case.arguments):
            digest.update(json.dumps([pos, list(array.shape), str(array.dtype)]).encode())
            if pos not in case.outputs or pos in case.inouts:
                digest.update(array.tobytes())
        for array in case.expected:
            digest.update(array.tobytes())
    return digest.hexdigest()
