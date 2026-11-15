"""Restricted simulation of real tvm.tir nodes; no script()/AST round trip.

TVM owns parsing and structural verification. This class owns the explicitly
listed execution semantics. Unknown nodes/calls/layouts stop, never disappear.
"""
from __future__ import annotations

import numpy as np

from .execution import Buffer, Budget, Expr, SemanticFault, Unsupported, apply, cast, select


class TVMInterpreter:
    binary = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div",
              "FloorDiv": "floordiv", "FloorMod": "mod", "Mod": "rem",
              "EQ": "eq", "NE": "ne", "LT": "lt", "LE": "le", "GT": "gt", "GE": "ge",
              "And": "and", "Or": "or", "Min": "min", "Max": "max"}
    calls = {"tir.exp": "exp", "tir.exp2": "exp2", "tir.log": "log", "tir.sqrt": "sqrt",
             "tir.rsqrt": "rsqrt", "tir.tanh": "tanh", "tir.fabs": "abs", "tir.sigmoid": "sigmoid"}

    def __init__(self, function, *, max_steps=2_000_000):
        from tilelang import tvm
        self.tvm, self.fun, self.max_steps = tvm, function, max_steps
        allowed = set(self.binary) | {'IntImm','FloatImm','StringImm','Var','SizeVar',
            'BufferLoad','Cast','Not','Select','Let','Call','Ramp','SeqStmt','LetStmt',
            'For','IfThenElse','BufferStore','Evaluate','AttrStmt','Block','BlockRealize'}
        calls = set(self.calls) | {'tir.if_then_else','tir.assume',
            'tl.tileop.region','tl.tileop.copy','tl.tileop.reduce'}
        gaps = []
        def preflight(node):
            kind = type(node).__name__
            if kind not in allowed:
                gaps.append(f'TIR node {kind} 尚未支持')
            if kind == 'Call' and getattr(node.op,'name',None) not in calls:
                gaps.append(f'TIR Call {getattr(node.op,"name",node.op)} 尚未支持（包括未执行分支）')
        tvm.tir.stmt_functor.post_order_visit(function.body,preflight)
        # Throw outside the FFI callback so TVM does not wrap Unsupported as an
        # internal TVMError and erase the supported/unsupported distinction.
        if gaps: raise Unsupported(gaps[0])

    def key(self, node):
        # TVM handles identify nodes; printed variable names need not be unique.
        return hash(node)

    def location(self, node, path):
        span = getattr(node, "span", None)
        value = {"node_path": path, "node_type": type(node).__name__, "stage_text_line": None}
        if span is not None:
            value["original_span"] = {"source": str(span.source_name.name), "line": int(span.line),
                                      "column": int(span.column), "end_line": int(span.end_line)}
        return value

    def event(self, node, path, **detail):
        self.events.append({**self.location(node, path), **detail})
        if len(self.events) > 128:
            self.events.pop(0)

    def expr(self, node, path):
        self.budget.tick()
        kind = type(node).__name__
        if kind in {"IntImm", "FloatImm"}:
            return cast(node.value, str(node.dtype))
        if kind == "StringImm":
            return str(node.value)
        if kind in {"Var", "SizeVar"}:
            if self.key(node) not in self.env:
                raise Unsupported(f"{path}: 未绑定变量 {node.name}")
            return self.env[self.key(node)]
        if kind == "Cast":
            return cast(self.expr(node.value, path+".value"), str(node.dtype))
        if kind in self.binary:
            if kind == 'Div' and not str(node.dtype).startswith('float'):
                raise Unsupported(f'{path}: 整数截断除法尚未支持；不经宿主浮点近似')
            values = [self.expr(node.a,path+'.a'),self.expr(node.b,path+'.b')]
            if kind in {'Min','Max'} and str(node.dtype).startswith('float'):
                if any(isinstance(v,Expr) or np.isnan(v) or (v==0 and np.signbit(v)) for v in values):
                    raise Unsupported(f'{path}: 浮点 Min/Max 的符号/NaN/负零规则尚未支持')
            return apply(self.binary[kind], values, str(node.dtype))
        if kind == "Not":
            return apply("not", [self.expr(node.a,path+".a")], "bool")
        if kind == "BufferLoad":
            if getattr(node, "predicate", None) is not None:
                raise Unsupported(f"{path}: predicated BufferLoad 未支持")
            if any(type(i).__name__ == "Ramp" for i in node.indices):
                raise Unsupported(f"{path}: vector load 仅在明确的 tile reduce 区域中支持")
            buf = self.buffers[self.key(node.buffer)]
            index = tuple(self.expr(i, path+f".indices[{j}]") for j,i in enumerate(node.indices))
            self.event(node,path,action="read",buffer=buf.name,index=[str(i) if isinstance(i,Expr) else int(i) for i in index])
            return buf.read(index,0)
        if kind == "Select":
            # Select evaluates both operands; tir.if_then_else below is lazy.
            return select(self.expr(node.condition,path+".condition"), self.expr(node.true_value,path+".true"), self.expr(node.false_value,path+".false"))
        if kind == "Let":
            key = self.key(node.var)
            old = self.env.get(key)
            self.env[key] = self.expr(node.value,path+".value")
            try:
                return self.expr(node.body,path+".body")
            finally:
                if old is None: self.env.pop(key,None)
                else: self.env[key] = old
        if kind == "Call":
            op = getattr(node.op,"name",None)
            if op == "tir.if_then_else" and len(node.args) == 3:
                condition = self.expr(node.args[0],path+".condition")
                if isinstance(condition,Expr):
                    return select(condition,self.expr(node.args[1],path+".true"),self.expr(node.args[2],path+".false"))
                branch = 1 if bool(condition) else 2
                return self.expr(node.args[branch],path+f".args[{branch}]")
            if op in self.calls:
                return apply(self.calls[op],[self.expr(a,path+f".args[{i}]") for i,a in enumerate(node.args)],str(node.dtype))
            raise Unsupported(f"{path}: 未支持 Call {op or node.op}")
        raise Unsupported(f"{path}: 未支持表达式 {kind}")

    def region(self, node, path, *, explicit=False):
        if type(node).__name__ != "BufferLoad":
            raise Unsupported(f"{path}: tile 区域必须是 BufferLoad/Ramp")
        if getattr(node,"predicate",None) is not None:
            raise Unsupported(f"{path}: predicated region 未支持")
        origin,shape = [],[]
        for i,index in enumerate(node.indices):
            if type(index).__name__ == "Ramp":
                if int(self.expr(index.stride,path)) != 1:
                    raise Unsupported(f"{path}: 仅支持单位步长归约区域")
                origin.append(int(self.expr(index.base,path)))
                shape.append(int(self.expr(index.lanes,path)))
            else:
                origin.append(int(self.expr(index,path+f".indices[{i}]")));shape.append(1)
        return self.buffers[self.key(node.buffer)],tuple(origin),tuple(shape)

    def tile_region(self, node, path):
        if type(node).__name__ != "Call" or getattr(node.op,"name",None) != "tl.tileop.region":
            raise Unsupported(f"{path}: copy 需要 tl.tileop.region")
        buf,origin,_ = self.region(node.args[0],path+".args[0]")
        shape = tuple(int(self.expr(a,path)) for a in node.args[2:])
        if len(shape) != len(buf.shape) or any(n<0 for n in shape):
            raise Unsupported(f"{path}: 非法 region rank/extent")
        self.budget.shape(shape)
        return buf,origin,shape

    def tile(self, node, path):
        op = getattr(node.op,"name",None)
        args = node.args
        if op == "tl.tileop.copy" and len(args) == 2:
            source, target = [self.tile_region(a,path+f".args[{i}]") for i,a in enumerate(args)]
            if source[0] is target[0]: raise Unsupported(f'{path}: copy 源目标别名尚未支持')
            sa = [i for i,n in enumerate(source[2]) if n != 1]
            ta = [i for i,n in enumerate(target[2]) if n != 1]
            shape = tuple(source[2][i] for i in sa)
            if shape != tuple(target[2][i] for i in ta):
                raise Unsupported(f"{path}: copy 非单位维度不一致；不猜 reshape/transpose")
            values = []
            for idx in np.ndindex(shape):
                a,b = list(source[1]),list(target[1])
                for axis,i in zip(sa,idx): a[axis] += i
                for axis,i in zip(ta,idx): b[axis] += i
                if not all(0<=i<n for i,n in zip(b,target[0].shape)): continue
                value = source[0].read(tuple(a),0) if all(0<=i<n for i,n in zip(a,source[0].shape)) else cast(0,str(target[0].dtype))
                values.append((tuple(b),value))
            # Actual RISC-V scalar copy pads invalid sources / masks destinations.
            for idx,value in values: target[0].write(idx,value,0)
            if target[0].position is not None:
                for idx,_ in values: self.writer_locations[(target[0].position,idx)] = self.location(node,path)
            self.event(node,path,action="tile_copy",source=source[0].name,destination=target[0].name)
            return
        if op == "tl.tileop.reduce" and len(args) == 5:
            source,origin,shape = self.region(args[0],path+".source")
            target,dest,dshape = self.region(args[1],path+".destination")
            kind,axis,clear = [self.expr(a,path+f".args[{i+2}]") for i,a in enumerate(args[2:])]
            axis = int(axis)
            if kind != "sum" or not 0<=axis<len(shape):
                raise Unsupported(f"{path}: 当前对象归约仅支持显式 sum 轴")
            out_shape = shape[:axis]+shape[axis+1:]
            if dshape != out_shape or source.dtype != target.dtype or source is target:
                raise Unsupported(f"{path}: reduce 区域、dtype 或别名不支持")
            self.reductions.append({**self.location(node,path),"kind":kind,"axis":axis,"clear":bool(clear),
                                    "order":"increasing_axis_index","accumulator_dtype":str(target.dtype)})
            for idx in np.ndindex(out_shape):
                dst = tuple(a+b for a,b in zip(dest,idx))
                value = cast(0,str(target.dtype)) if clear else target.read(dst,0)
                for k in range(shape[axis]):
                    index = idx[:axis]+(k,)+idx[axis:]
                    item = source.read(tuple(a+b for a,b in zip(origin,index)),0)
                    value = apply("add",[value,item],str(target.dtype))
                target.write(dst,value,0)
            self.event(node,path,action="tile_reduce",source=source.name,destination=target.name)
            return
        if op == "tir.assume" and len(args) == 1:
            if not bool(self.expr(args[0],path)): raise Unsupported(f"{path}: 输入不满足 assume")
            return
        raise Unsupported(f"{path}: 未支持 Evaluate Call {op}")

    def alloc(self, node, path):
        from .execution import dtype
        if len(node.strides) or len(node.axis_separators) or int(self.expr(node.elem_offset,path)) != 0:
            raise Unsupported(f"{path}: 局部显式 stride/offset 尚未支持")
        shape = tuple(int(self.expr(s,path)) for s in node.shape)
        self.budget.shape(shape)
        self.buffers[self.key(node)] = Buffer(np.empty(shape,dtype=dtype(str(node.dtype))),initialized=False,
            symbolic=self.symbolic,budget=self.budget,name=str(node.name))

    def uses(self, node, variable):
        # Consider executable expressions, not declarative reads/writes metadata.
        kind = type(node).__name__
        if kind == "Block":
            return self.uses(node.body,variable) or (node.init is not None and self.uses(node.init,variable))
        if kind == "BlockRealize":
            return self.uses(node.block,variable) or self.uses(node.predicate,variable)
        if kind == "AttrStmt": return self.uses(node.body,variable) or self.uses(node.value,variable)
        if kind == "SeqStmt": return any(self.uses(s,variable) for s in node.seq)
        found = []
        self.tvm.tir.stmt_functor.post_order_visit(node,lambda n: found.append(True) if isinstance(n,self.tvm.tir.Var) and n.same_as(variable) else None)
        return bool(found)

    def stmt(self, node, path="body"):
        self.budget.tick()
        try:
            with self.budget.at("tvm_node",0,**self.location(node,path)):
                return self._stmt(node,path)
        except SemanticFault as exc:
            exc.line = None
            exc.details.setdefault("tvm_location",self.location(node,path))
            raise

    def _stmt(self, node, path):
        kind = type(node).__name__
        if kind == "SeqStmt":
            for i,s in enumerate(node.seq): self.stmt(s,path+f".seq[{i}]")
        elif kind == "LetStmt":
            key = self.key(node.var);old = self.env.get(key)
            self.env[key] = self.expr(node.value,path+".value")
            try: self.stmt(node.body,path+".body")
            finally:
                if old is None: self.env.pop(key,None)
                else: self.env[key] = old
        elif kind == "For":
            # Serial, Parallel, Unrolled only; vectorized/thread bindings need rules.
            loop_kind = int(node.kind)
            if loop_kind not in {0,1,3}:
                raise Unsupported(f"{path}: For kind={node.kind} 尚未支持")
            if len(node.annotations):
                raise Unsupported(f"{path}: 带布局/其他 annotations 的 For 尚未支持")
            lo,n = int(self.expr(node.min,path+".min")),int(self.expr(node.extent,path+".extent"))
            if n<0: raise Unsupported(f"{path}: negative extent")
            region = self.budget.new_region() if loop_kind == 1 else None
            key = self.key(node.loop_var)
            old = self.loop_specs.get(key)
            self.loop_specs[key] = (lo,n,loop_kind)
            try:
                for i in range(lo,lo+n):
                    self.env[key] = cast(i,str(node.loop_var.dtype))
                    if region is None: self.stmt(node.body,path+".body")
                    else:
                        with self.budget.iteration(region,i): self.stmt(node.body,path+".body")
            finally:
                if old is None: self.loop_specs.pop(key,None)
                else: self.loop_specs[key] = old
        elif kind == "IfThenElse":
            cond = self.expr(node.condition,path+".condition")
            if isinstance(cond,Expr): raise Unsupported(f"{path}: 数据相关符号分支尚未支持；可执行具体输入")
            branch = node.then_case if bool(cond) else node.else_case
            if branch is not None: self.stmt(branch,path+(".then" if bool(cond) else ".else"))
        elif kind == "BufferStore":
            if getattr(node,"predicate",None) is not None: raise Unsupported(f"{path}: predicated BufferStore 未支持")
            buf = self.buffers[self.key(node.buffer)]
            idx = tuple(self.expr(i,path+f".indices[{j}]") for j,i in enumerate(node.indices))
            value = self.expr(node.value,path+".value")
            buf.write(idx,value,0)
            if buf.position is not None:
                self.writer_locations[(buf.position,tuple(map(int,idx)))] = self.location(node,path)
            self.event(node,path,action="write",buffer=buf.name,index=[int(i) for i in idx])
        elif kind == "Evaluate":
            if type(node.value).__name__ == "Call": self.tile(node.value,path+".value")
            elif type(node.value).__name__ == "IntImm" and int(node.value.value) == 0: pass
            else: raise Unsupported(f"{path}: 未支持 Evaluate {type(node.value).__name__}")
        elif kind == "AttrStmt":
            if str(node.attr_key) != "thread_extent":
                raise Unsupported(f"{path}: 未支持 AttrStmt {node.attr_key}")
            var,tag = node.node.var,str(node.node.thread_tag)
            extent = int(self.expr(node.value,path+".value"))
            if extent<1: raise Unsupported(f"{path}: 非法 thread extent")
            if tag.startswith("blockIdx."):
                for i in range(extent):
                    self.env[self.key(var)] = cast(i,str(var.dtype));self.stmt(node.body,path+".body")
            elif tag.startswith("threadIdx.") and (extent == 1 or not self.uses(node.body,var)):
                self.env[self.key(var)] = cast(0,str(var.dtype));self.stmt(node.body,path+".body")
            else:
                raise Unsupported(f"{path}: 使用了物理线程轴 {tag}；当前对象执行器不猜线程映射")
        elif kind == "BlockRealize":
            if not bool(self.expr(node.predicate,path+".predicate")): return
            block = node.block
            if len(block.iter_vars) != len(node.iter_values): raise Unsupported(f"{path}: Block iter mismatch")
            for var,val in zip(block.iter_vars,node.iter_values):
                self.env[self.key(var.var)] = self.expr(val,path+".iter_value")
            if block.init is None:
                self.stmt(block,path+".block")
            else:
                # Standard unscheduled reduction: a direct, complete serial
                # domain. Scheduled/rfactor/predicated forms need other rules.
                if (type(node.predicate).__name__ != 'IntImm' or not int(node.predicate.value)
                        or len(block.alloc_buffers) or len(block.match_buffers) or len(block.annotations)):
                    raise Unsupported(f'{path}: reduction init 仅支持无谓词、无局部重分配的标准块')
                first = [];axes = [];bound_axes = set()
                for iv,actual in zip(block.iter_vars,node.iter_values):
                    axis_kind = int(iv.iter_type)
                    if axis_kind not in {0,2} or type(actual).__name__ != 'Var' or iv.dom is None:
                        raise Unsupported(f'{path}: reduction init 需要直接的串行归约轴映射')
                    lo = int(self.expr(iv.dom.min,path+'.domain.min'))
                    size = int(self.expr(iv.dom.extent,path+'.domain.extent'))
                    key = self.key(actual)
                    if key in bound_axes:
                        raise Unsupported(f'{path}: 块轴映射必须独立，不能共用同一循环轴')
                    bound_axes.add(key)
                    if axis_kind == 0:
                        spec = self.loop_specs.get(key)
                        if spec is None or spec[:2] != (lo,size):
                            raise Unsupported(f'{path}: 空间轴必须直接对应完整循环域')
                        continue
                    if self.loop_specs.get(self.key(actual)) != (lo,size,0):
                        raise Unsupported(f'{path}: 归约循环必须完整、串行且与 Block 域一致')
                    if self.uses(block.init,iv.var) or self.uses(block.init,actual):
                        raise Unsupported(f'{path}: 初始化依赖归约轴，尚未建立执行契约')
                    first.append(int(self.env[self.key(iv.var)]) == lo)
                    axes.append({'variable':str(iv.var.name),'min':lo,'extent':size})
                if not axes: raise Unsupported(f'{path}: init 块缺少显式归约轴')
                initialize = all(first)
                if initialize:
                    self.stmt(block.init,path+'.block.init')
                    self.reductions.append({**self.location(block,path+'.block'),
                        'kind':'block_init','axes':axes,'initialization':'first_reduction_iteration',
                        'order':'original serial loop nesting'})
                self.stmt(block.body,path+'.block.body')
        elif kind == "Block":
            if node.init is not None or len(node.match_buffers) or len(node.annotations):
                raise Unsupported(f"{path}: Block init/match_buffers/annotations 尚未支持")
            for i,buf in enumerate(node.alloc_buffers): self.alloc(buf,path+f".alloc_buffers[{i}]")
            self.stmt(node.body,path+".body")
        else:
            raise Unsupported(f"{path}: 未支持语句 {kind}")

    def run(self, buffers, symbols=None, symbolic=False):
        from .execution import dtype
        self.budget = buffers[0].budget if buffers else Budget(self.max_steps)
        self.symbolic,self.env,self.buffers,self.events,self.reductions = symbolic,{},{},[],[]
        self.writer_locations = {}
        self.loop_specs = {}
        if len(self.fun.params) != len(buffers): raise Unsupported("TIR 参数数量与测试契约不同")
        for param,buf in zip(self.fun.params,buffers):
            if param not in self.fun.buffer_map: raise Unsupported("目前只支持 buffer 参数")
            spec = self.fun.buffer_map[param]
            if len(spec.axis_separators): raise Unsupported('带 axis_separators 的 buffer 尚未支持')
            if dtype(str(spec.dtype)) != buf.dtype or len(spec.shape) != len(buf.shape):
                raise Unsupported("TIR buffer dtype/rank 与测试契约不同")
            for dimension,size in zip(spec.shape,buf.shape):
                if type(dimension).__name__ in {"Var","SizeVar"}:
                    key = self.key(dimension)
                    previous = self.env.setdefault(key,cast(size,str(dimension.dtype)))
                    if int(previous) != size: raise Unsupported("共享动态 shape 符号不一致")
            self.buffers[self.key(spec)] = buf
        for param,buf in zip(self.fun.params,buffers):
            spec = self.fun.buffer_map[param]
            if tuple(int(self.expr(s,"buffer.shape")) for s in spec.shape) != buf.shape:
                raise Unsupported("TIR shape 与用例不一致；当前只直接绑定符号维度，不求解复合形状表达式")
            if int(self.expr(spec.elem_offset,"buffer.elem_offset")) != 0:
                raise Unsupported("非零 elem_offset 尚未支持")
            actual = tuple(i//buf.dtype.itemsize for i in buf.values.strides)
            if len(spec.strides) and tuple(int(self.expr(s,"buffer.strides")) for s in spec.strides) != actual:
                raise Unsupported("显式 stride 与用例连续布局不一致")
        self.stmt(self.fun.body)
        return {"steps":self.budget.steps,"backend":"tvm_tir_nodes",
                "execution_model":"logical blocks; only unused physical thread axes",
                "location_model":"node paths + original spans; no exported-text line mapping",
                "node_events":self.events,"reductions":self.reductions}
