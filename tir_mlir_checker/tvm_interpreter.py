"""Restricted simulation of real tvm.tir nodes; no script()/AST round trip.

TVM owns parsing and structural verification. This class owns the explicitly
listed execution semantics. Unknown nodes/calls/layouts stop, never disappear.
"""
from __future__ import annotations

import numpy as np

from .capabilities import (TIR_BATCH2_TILES,TIR_BATCH_SCALARS,TIR_BREAKS,TIR_CUMSUM,TIR_EXTERN,
                           TIR_SERIAL_MARKERS,TIR_TAIL_STATEMENTS,TIR_TUPLE)
from .execution import Buffer, Budget, Expr, LoopBreak, SemanticFault, Unsupported, apply, cast, concrete_index, extern_scalar, select, tir_scalar
from .tir_tail_contracts import SWIZZLE_ATTR, WARP_FENCE, extern_error, marker_error, tuple_error
from .tile_ops import (GEMM_OPS, accumulate_matrix, check_hints, copy_region, cumsum_region, fill_region,
                       gemm_controls, gemm_view, matmul_shape, matrix_step, reduction_details,
                       reduce_region, region_descriptor)


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
            'For','IfThenElse','BufferStore','Evaluate','AttrStmt','Block','BlockRealize','BufferRegion','Range'}
        calls = set(self.calls) | set(TIR_BATCH_SCALARS) | TIR_BATCH2_TILES | TIR_TAIL_STATEMENTS | {TIR_EXTERN,'tir.if_then_else','tir.assume',
            'tl.tileop.region','tl.tileop.copy','tl.tileop.reduce'}
        gaps,tail_nodes,standalone=set(),{},set()
        def preflight(node):
            kind = type(node).__name__
            if kind not in allowed:
                gaps.add(f'TIR node {kind} 尚未支持')
            op = self.op_name(node)
            if kind == 'Call' and op not in calls:
                gaps.add(f'TIR Call {op or node.op} 尚未支持（包括未执行分支）')
            if op in TIR_TAIL_STATEMENTS or op==TIR_EXTERN:
                problem=self.tail_error(node,op)
                if problem: gaps.add(problem)
                if op in TIR_TAIL_STATEMENTS: tail_nodes[self.key(node)]=op
            if kind == 'Block':
                try:
                    self.check_block_metadata(node,'Block')
                except Unsupported as exc:
                    gaps.add(str(exc))
            if kind=='Evaluate' and self.op_name(node.value) in TIR_TAIL_STATEMENTS:
                standalone.add(self.key(node.value))
            if kind=='AttrStmt' and str(node.attr_key)==SWIZZLE_ATTR and self.op_name(node.value)==TIR_TUPLE:
                standalone.add(self.key(node.value))
                problem=self.tail_error(node.value,TIR_TUPLE,metadata=True)
                if problem: gaps.add(problem)
            # Also reject reuse of the same Call object in a value context.
            scalar_fields={
                'BufferLoad':('indices','predicate'),'BufferStore':('value','indices','predicate'),
                'For':('min','extent'),'IfThenElse':('condition',),'BlockRealize':('iter_values','predicate'),
                'Select':('condition','true_value','false_value'),'Let':('value','body'),'LetStmt':('value',),
                'Cast':('value',),'Not':('a',),'Ramp':('base','stride','lanes'),'Call':('args',),
                'Range':('min','extent'),
            }.get(kind,('a','b') if kind in self.binary else ())
            for field in scalar_fields:
                value=getattr(node,field,None)
                vals=value if field in {'args','indices','iter_values'} and value is not None else (value,)
                for v in vals:
                    if self.op_name(v) in TIR_TAIL_STATEMENTS:
                        gaps.add(f'{self.op_name(v)}: 仅支持 Evaluate 或允许的元数据，不能用作值')
            if op in TIR_BATCH2_TILES:
                if len(node.args) not in ((10,19) if op in GEMM_OPS else (2,)):
                    gaps.add(f'{op}: 不支持的位置参数数量')
                try:
                    self.tile_hints(node,op)
                except Unsupported as exc:
                    gaps.add(str(exc))
        tvm.tir.stmt_functor.post_order_visit(function.body,preflight)
        for key,op in tail_nodes.items():
            if key not in standalone: gaps.add(f'{op}: 仅支持 Evaluate 或允许的元数据，不能用作值')
        # Throw outside the FFI callback so TVM does not wrap Unsupported as an
        # internal TVMError and erase the supported/unsupported distinction.
        if gaps: raise Unsupported(sorted(gaps)[0])

    def key(self, node):
        # TVM handles identify nodes; printed variable names need not be unique.
        return hash(node)

    @staticmethod
    def op_name(node):
        return getattr(node.op,'name',None) if type(node).__name__=='Call' else None

    @staticmethod
    def constant(node):
        kind=type(node).__name__
        if kind=='StringImm': return str(node.value)
        if kind=='FloatImm': return float(node.value)
        if kind=='IntImm':
            dt=str(node.dtype)
            return bool(node.value) if dt=='bool' else int(node.value) if dt.startswith(('int','uint')) and 'x' not in dt else None
        return None

    def tail_error(self,node,op,metadata=False):
        if getattr(node,'annotations',None):
            return f'{op}: 不支持附加注解'
        args=node.args
        vals=[self.constant(a) for a in args]
        if op==TIR_EXTERN:
            return extern_error(vals,str(node.dtype),[str(getattr(a,'dtype','')) for a in args[1:]])
        if op in TIR_BREAKS and args:
            return f'{op}: 不接受参数'
        if op==TIR_CUMSUM and (len(args)!=4 or type(vals[2]) is not int or
                               type(vals[3]) not in (bool,int) or vals[3] not in (0,1)):
            return 'cumsum 需要 4 参数和静态整数 dim / 布尔 reverse'
        if op==TIR_TUPLE:
            return tuple_error(vals,metadata)
        if op in TIR_SERIAL_MARKERS:
            pointer=(op==WARP_FENCE and len(args)==4 and type(args[1]).__name__=='Var'
                     and str(args[1].dtype)=='handle')
            return marker_error(op,vals,pointer=pointer)
        return None

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
            if op==TIR_EXTERN:
                problem=self.tail_error(node,op)
                if problem: raise Unsupported(problem)
                values=[self.expr(a,path+f'.args[{i}]') for i,a in enumerate(node.args[1:],1)]
                return extern_scalar(self.constant(node.args[0]),values,str(node.dtype),
                                     [str(a.dtype) for a in node.args[1:]])
            if op == "tir.if_then_else" and len(node.args) == 3:
                condition = self.expr(node.args[0],path+".condition")
                if isinstance(condition,Expr):
                    return select(condition,self.expr(node.args[1],path+".true"),self.expr(node.args[2],path+".false"))
                branch = 1 if bool(condition) else 2
                return self.expr(node.args[branch],path+f".args[{branch}]")
            if op in TIR_BATCH_SCALARS:
                values = [self.expr(a, path+f".args[{i}]") for i,a in enumerate(node.args)]
                types = [str(a.dtype) for a in node.args]
                try:
                    return tir_scalar(op, values, str(node.dtype), types)
                except SemanticFault as exc:
                    exc.details.setdefault("tvm_location", self.location(node, path))
                    raise
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
                if type(index.stride).__name__ != 'IntImm' or int(index.stride.value) != 1:
                    raise Unsupported(f"{path}: 仅支持单位步长归约区域")
                if type(index.lanes).__name__ != 'IntImm':
                    raise Unsupported(f'{path}: TIR region 需要静态 Ramp lanes')
                origin.append(concrete_index(self.expr(index.base,path)))
                shape.append(int(index.lanes.value))
            else:
                origin.append(concrete_index(self.expr(index,path+f".indices[{i}]")));shape.append(1)
        return self.buffers[self.key(node.buffer)],tuple(origin),tuple(shape)

    def tile_region(self, node, path):
        kind = type(node).__name__
        if kind == 'BufferLoad':
            region = region_descriptor(*self.region(node,path))
            return region, tuple(i for i,n in enumerate(region[2]) if n==1)
        if kind == 'BufferRegion':
            buf = self.buffers[self.key(node.buffer)]
            origin = tuple(self.expr(r.min,path) for r in node.region)
            extents = [r.extent for r in node.region]
        elif kind == 'Call' and getattr(node.op,'name',None) == 'tl.tileop.region' and len(node.args)>=2:
            load = node.args[0]
            buf,origin,lane_sizes = self.region(load,path+'.args[0]')
            if type(node.args[1]).__name__ != 'IntImm' or int(node.args[1].value) not in range(1,4):
                raise Unsupported(f'{path}: 不支持的 region access mask')
            extents = tuple(i.lanes if type(i).__name__=='Ramp' else provided
                            for i,provided in zip(load.indices,node.args[2:]))
            if len(load.indices)!=len(node.args)-2 or any(
                    type(i).__name__=='Ramp' and type(n).__name__=='IntImm'
                    and int(n.value)!=lanes for i,n,lanes in zip(load.indices,node.args[2:],lane_sizes)):
                raise Unsupported(f'{path}: region rank 或 Ramp lanes 与显式 extent 不符')
        else:
            raise Unsupported(f'{path}: 需要 tl.tileop.region、BufferRegion 或 BufferLoad/Ramp')
        shape = tuple(self.expr(r,path) for r in extents)
        ones = tuple(i for i,n in enumerate(extents) if type(n).__name__=='IntImm' and int(n.value)==1)
        return region_descriptor(buf,origin,shape),ones

    def tile_hints(self,node,op):
        hints = {}
        annotations = getattr(node,'annotations',None)
        if annotations is not None:
            for key,value in annotations.items():
                hints[str(key)] = value.value if type(value).__name__ in {'IntImm','StringImm'} else value
        check_hints(op,hints)

    def tile(self, node, path):
        op = getattr(node.op,"name",None)
        args = node.args
        def write(buf, index, value, line):
            buf.write(index,value,line)
            if buf.position is not None:
                self.writer_locations[(buf.position,buf.root_index(index))] = self.location(node,path)

        if op in TIR_TAIL_STATEMENTS:
            problem=self.tail_error(node,op)
            if problem: raise Unsupported(problem)
            if op in TIR_BREAKS:
                if not self.loop_stack or self.loop_stack[-1]!=0:
                    raise Unsupported(f'{op}: 只能跳出最内层串行 For')
                raise LoopBreak()
            if op in TIR_SERIAL_MARKERS or op==TIR_TUPLE:
                self.event(node,path,action='serial_noop',operation=op)
                return
            if op==TIR_CUMSUM:
                if self.nonunit_threads: raise Unsupported('cumsum 仅支持非协作、串行区域')
                dim,reverse=map(self.constant,args[2:])
                cumsum_region(*(self.tile_region(a,path+f'.args[{i}]')[0] for i,a in enumerate(args[:2])),
                              dim,bool(reverse),0,write,
                              lambda details:self.reductions.append({**self.location(node,path),**details}))
                self.event(node,path,action='tile_cumsum',operation=op)
                return
        if op==TIR_EXTERN:
            self.expr(node,path)
            return

        if op in {'tl.tileop.copy','tl.tileop.async_copy','tl.tileop.tma_copy'}:
            if len(args)!=2:
                raise Unsupported(f'{op} 只接受两个位置参数；调度提示必须是已知注解')
            if op != 'tl.tileop.copy': self.tile_hints(node,op)
            source,target = [self.tile_region(a,path+f'.args[{i}]')[0] for i,a in enumerate(args)]
            copy_region(source,target,0,write)
            self.event(node,path,action='tile_copy',operation=op,source=source[0].name,destination=target[0].name)
            return
        if op == 'tl.tileop.fill' and len(args)==2:
            self.tile_hints(node,op)
            dst,_ = self.tile_region(args[0],path+'.args[0]')
            fill_region(dst,self.expr(args[1],path+'.args[1]'),0,write)
            self.event(node,path,action='tile_fill',destination=dst[0].name)
            return
        if op in GEMM_OPS and len(args) in (10,19):
            self.tile_hints(node,op)
            if any(type(args[i]).__name__ != 'IntImm' for i in (3,4,8)):
                raise Unsupported(f'{path}: GEMM transpose 与 warp policy 必须静态')
            if len(args)==19 and (type(args[14]).__name__!='IntImm' or
                                  type(args[15]).__name__!='IntImm' or
                                  type(args[16]).__name__!='IntImm'):
                raise Unsupported(f'{path}: GEMM k_pack/wg_wait 必须静态且不支持 mbar')
            regions,ones = zip(*(self.tile_region(arg,path+f'.args[{i}]') for i,arg in enumerate(args[:3])))
            a,b,c = (gemm_view(reg,drops) for reg,drops in zip(regions,ones))
            ta,tb,dims,clear = gemm_controls([self.expr(arg,path+f'.args[{i}]')
                                            for i,arg in enumerate(args[3:],3)],regions)
            shape = matmul_shape(a,b,c,ta,tb,dims)
            step = matrix_step(a,b,c)
            self.reductions.append({**self.location(node,path),**reduction_details(c,ta,tb,clear)})
            if clear: fill_region(regions[2],0,0,write)
            accumulate_matrix(a,b,c,ta,tb,shape,0,step,write)
            self.event(node,path,action='tile_gemm',operation=op,source=(a.name,b.name),destination=c.name)
            return
        if op == "tl.tileop.reduce" and len(args) == 5:
            source,target = (self.region(a,path+f'.args[{i}]') for i,a in enumerate(args[:2]))
            kind,axis,clear = map(self.constant,args[2:])
            if type(kind) is not str or type(axis) is not int or type(clear) not in (bool,int):
                raise Unsupported(f'{path}: reduce 的 kind / dim / clear 必须是字面量')
            reduce_region(source,target,kind,axis,clear,0,write,
                          lambda info:self.reductions.append({**self.location(node,path),**info}))
            self.event(node,path,action='tile_reduce',source=source[0].name,destination=target[0].name)
            return
        if op == "tir.assume" and len(args) == 1:
            if not bool(self.expr(args[0],path)): raise Unsupported(f"{path}: 输入不满足 assume")
            return
        raise Unsupported(f"{path}: 未支持 Evaluate Call {op}")

    def check_block_metadata(self, block, path):
        if len(block.match_buffers):
            raise Unsupported(f'{path}: Block match_buffers 尚未支持')
        if not len(block.annotations):
            return
        from tilelang.layout import Layout

        for key, value in block.annotations.items():
            if str(key) != 'layout_map':
                raise Unsupported(f'{path}: 未支持 Block 注解 {key}')
            if (block.init is not None or len(block.iter_vars) or
                    type(value).__name__ != 'Map'):
                raise Unsupported(f'{path}: layout_map 仅用于无 init、无迭代变量的局部块')
            buffers = {self.key(buf.data):buf for buf in block.alloc_buffers}
            for var,layout in value.items():
                buf = buffers.get(self.key(var))
                if (buf is None or not buf.data.same_as(var) or buf.scope() != 'shared.dyn'
                        or len(buf.shape) != 2 or len(buf.strides) or len(buf.axis_separators)
                        or self.constant(buf.elem_offset) != 0 or type(layout) is not Layout):
                    raise Unsupported(f'{path}: layout_map 需要块内二维连续 shared.dyn 分配和普通 Layout')
                shape = tuple(map(self.constant,buf.shape))
                declared = tuple(map(self.constant,layout.get_input_shape()))
                if any(type(n) is not int or n <= 0 for n in shape) or declared != shape:
                    raise Unsupported(f'{path}: layout_map 输入尺寸须与局部分配一致')
                coords,forward = list(layout.get_forward_vars()),list(layout.get_forward_index())
                # The captured tiling uses only injective row-major packing,
                # possibly with row padding. Logical coordinates ignore the
                # padding; no two coordinates may alias. Swizzles, fragments
                # and mappings to a different buffer are outside this contract.
                if len(coords) != 2 or len(forward) != 1:
                    raise Unsupported(f'{path}: layout_map 仅支持二维行主序 packing')
                def placeholder(value, pos):
                    expected = coords[pos]
                    # Layout's input placeholders are synthetic and scoped to
                    # this descriptor. JSON can recreate their object handles.
                    return (type(value).__name__ == 'Var'
                            and str(value.name) == str(expected.name)
                            and str(value.dtype) == str(expected.dtype))

                expr = forward[0]
                if type(expr).__name__ != 'Add' or not placeholder(expr.b,1):
                    raise Unsupported(f'{path}: layout_map 仅支持 row * stride + col')
                mul = expr.a
                stride = self.constant(mul.b) if type(mul).__name__ == 'Mul' and placeholder(mul.a,0) else None
                if type(stride) is not int or stride < shape[1]:
                    raise Unsupported(f'{path}: layout_map 行主序步长无效或产生重叠')
                max_index = (shape[0]-1)*stride + shape[1]-1
                index_dtype = str(expr.dtype)
                if index_dtype not in ('int32','int64') or max_index >= 1 << (int(index_dtype[3:])-1):
                    raise Unsupported(f'{path}: layout_map 物理索引类型非法或溢出')

    def check_for_metadata(self, node, path):
        if not len(node.annotations):
            return
        # One pipeline stage preserves the serial loop order in LowerLoopBody.
        # No dependence on pipeline prologues or overlapping stages is modeled.
        if (int(node.kind) == 0 and len(node.annotations) == 1
                and 'num_stages' in node.annotations
                and self.constant(node.annotations['num_stages']) == 1
                and not isinstance(self.constant(node.annotations['num_stages']),bool)
                and (getattr(node,'step',None) is None or self.constant(node.step)==1)):
            return
        raise Unsupported(f'{path}: 带布局/其他 annotations 的 For 尚未支持')

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
            self.check_for_metadata(node,path)
            lo,n = int(self.expr(node.min,path+".min")),int(self.expr(node.extent,path+".extent"))
            if n<0: raise Unsupported(f"{path}: negative extent")
            region = self.budget.new_region() if loop_kind == 1 else None
            key = self.key(node.loop_var)
            old = self.loop_specs.get(key)
            missing=object()
            old_value=self.env.get(key,missing)
            self.loop_specs[key] = (lo,n,loop_kind)
            self.loop_stack.append(loop_kind)
            try:
                for i in range(lo,lo+n):
                    self.env[key] = cast(i,str(node.loop_var.dtype))
                    if region is None: self.stmt(node.body,path+".body")
                    else:
                        with self.budget.iteration(region,i): self.stmt(node.body,path+".body")
            except LoopBreak:
                if loop_kind!=0: raise
            finally:
                self.loop_stack.pop()
                if old_value is missing: self.env.pop(key,None)
                else: self.env[key]=old_value
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
            if str(node.attr_key)==SWIZZLE_ATTR and self.op_name(node.value)==TIR_TUPLE:
                problem=self.tail_error(node.value,TIR_TUPLE,metadata=True)
                if problem: raise Unsupported(problem)
                self.event(node.value,path+'.value',action='serial_noop',operation=TIR_TUPLE)
                self.stmt(node.body,path+'.body')
                return
            if str(node.attr_key) != "thread_extent":
                raise Unsupported(f"{path}: 未支持 AttrStmt {node.attr_key}")
            var,tag = node.node.var,str(node.node.thread_tag)
            extent = int(self.expr(node.value,path+".value"))
            if extent<1: raise Unsupported(f"{path}: 非法 thread extent")
            if tag.startswith("blockIdx."):
                for i in range(extent):
                    self.env[self.key(var)] = cast(i,str(var.dtype));self.stmt(node.body,path+".body")
            elif tag.startswith("threadIdx.") and (extent == 1 or not self.uses(node.body,var)):
                self.env[self.key(var)] = cast(0,str(var.dtype))
                self.nonunit_threads += extent!=1
                try: self.stmt(node.body,path+".body")
                finally: self.nonunit_threads -= extent!=1
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
            if node.init is not None:
                raise Unsupported(f"{path}: 独立 Block.init 尚未支持")
            self.check_block_metadata(node,path)
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
        self.loop_stack,self.nonunit_threads = [],0
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
