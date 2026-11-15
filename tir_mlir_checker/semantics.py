"""Finite-shape symbolic comparison and memory evidence, no operator registry."""
import numpy as np

from .execution import Expr, expression_digest, provenance


def compare_symbolic(left,right):
    mask=np.zeros(left.shape,dtype=bool)
    first=None
    for index in np.ndindex(left.shape):
        a,b=left.expressions[index],right.expressions[index]
        equal=a==b
        mask[index]=equal
        if not equal and first is None:
            first={"index":list(index),"left":a.data() if a is not None else None,
                   "right":b.data() if b is not None else None,
                   "left_writer_line":int(left.last_lines[index]),
                   "right_writer_line":int(right.last_lines[index]),
                   "left_trace":provenance(a), "right_trace":provenance(b)}
    return {"status":"match" if bool(mask.all()) else "uncertain",
            "equal_expressions":int(mask.sum()),"total_expressions":int(mask.size),
            "left_digest":expression_digest(left.expressions),"right_digest":expression_digest(right.expressions),
            "first_difference":first,
            "scope":"对当前有限形状逐元素构造的表达式一致性；不同表达式不自动等于数学不等价。"}


def coverage(buffer):
    result=buffer.summary()
    if not getattr(buffer,"require_full_write",True):
        result["status"]="match"
        result["scope"]="显式 inout/工作区契约保留初值，不要求每个位置重写；仍比较完整输出值。"
        return result
    result["status"]="match" if result["unwritten"]==0 else "mismatch"
    if result["unwritten"]:
        result["first_unwritten_index"]=[int(v) for v in np.argwhere(buffer.writes==0)[0]]
    result["scope"]="当前测试形状、该后端执行中的输出写入覆盖；多次写入可能是合法累加，不自动判错。"
    return result
