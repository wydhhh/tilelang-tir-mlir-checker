"""Bounded execution of official MLIR objects; never parses printed assembly."""
from __future__ import annotations

import itertools
import re
import numpy as np

from .execution import (Buffer,Budget,Expr,SemanticFault,Unsupported,apply,cast,dtype,
                        literal,merge_trace,select,storage_type,trace_value,mlir_cast,mlir_cmpi)


class MLIRObjectInterpreter:
    binary = {'arith.addi':'add','arith.subi':'sub','arith.muli':'mul',
              'arith.addf':'add','arith.subf':'sub','arith.mulf':'mul','arith.divf':'div',
              'arith.andi':'and','arith.ori':'or','arith.floordivsi':'floordiv','arith.remsi':'rem'}
    unary = {'arith.negf':'neg','math.exp':'exp','math.exp2':'exp2','math.rsqrt':'rsqrt',
             'math.sqrt':'sqrt','math.log':'log','math.absf':'abs','math.tanh':'tanh'}
    casts = {'arith.index_cast','arith.index_castui','arith.extf','arith.truncf',
             'arith.sitofp','arith.uitofp','arith.extsi','arith.extui','arith.trunci'}

    def __init__(self,loaded,source,*,max_steps=2_000_000):
        import mlir.ir as ir
        self.ir,self.context,self.module,self.source,self.max_steps = ir,*loaded,source,max_steps
        self.lines = source.splitlines()
        # Explicit locations may refer to another file, or to stale source lines.
        self.text_locations = re.search(r'\bloc\s*\(',source) is None
        if len(self.module.operation.attributes):
            raise Unsupported('module 属性（含目标/data layout）尚未建立模拟契约')
        children = list(self.module.body.operations)
        if len(children)!=1 or children[0].operation.name!='func.func':
            raise Unsupported('官方 MLIR 模拟需要单个 func.func；不猜入口或外部调用')
        self.function = children[0].operation
        if len(self.function.regions)!=1 or len(self.function.regions[0].blocks)!=1:
            raise Unsupported('需要单个函数入口 block')
        self.paths = {}
        allowed = set(self.binary)|set(self.unary)|self.casts|{
            'func.func','func.return','arith.constant','arith.cmpi','arith.cmpf','arith.select',
            'memref.alloc','memref.alloca','memref.load','memref.store','memref.dim',
            'memref.copy','memref.subview','scf.for','scf.parallel','scf.if','scf.yield',
            'scf.reduce','linalg.fill','linalg.yield'}
        with self.context:
            def inspect(op,path):
                op=op.operation;self.paths[op]=path
                if op.name not in allowed: raise Unsupported(f'{path}: 未支持官方 MLIR 操作 {op.name}（包括未执行分支）')
                attrs=set(op.attributes)
                defaults={'overflowFlags':'#arith.overflow<none>','fastmath':'#arith.fastmath<none>'}
                for key,value in defaults.items():
                    if key in attrs and str(op.attributes[key])==value: attrs.remove(key)
                permitted={'func.func':{'sym_name','function_type','sym_visibility'},
                    'arith.constant':{'value'},'arith.cmpi':{'predicate'},'arith.cmpf':{'predicate'},
                    'scf.parallel':{'operandSegmentSizes','operand_segment_sizes'},
                    'linalg.fill':{'operandSegmentSizes','operand_segment_sizes'},
                    'memref.alloc':{'operandSegmentSizes','operand_segment_sizes','alignment'},
                    'memref.alloca':{'operandSegmentSizes','operand_segment_sizes','alignment'},
                    'memref.subview':{'static_offsets','static_sizes','static_strides','operandSegmentSizes','operand_segment_sizes'}}.get(op.name,set())
                if attrs-permitted:
                    raise Unsupported(f'{path}: {op.name} 未支持属性 {sorted(attrs-permitted)}')
                if op.name=='scf.parallel' and (len(op.results) or len(op.operands)!=3*len(op.regions[0].blocks[0].arguments)):
                    raise Unsupported(f'{path}: 带归约结果的 scf.parallel 尚未支持')
                if op.name=='scf.reduce' and (len(op.operands) or len(op.regions)):
                    raise Unsupported(f'{path}: 非空 scf.reduce 尚未支持')
                for ri,r in enumerate(op.regions):
                    if len(r.blocks)>1: raise Unsupported(f'{path}: 多 block/CFG 尚未支持')
                    for bi,b in enumerate(r.blocks):
                        for oi,child in enumerate(b.operations): inspect(child,f'{path}.r{ri}.b{bi}.op{oi}')
            inspect(self.function,'function')

    def location(self,op):
        loc=str(op.location);line=None
        try: parsed_file,parsed_line=op.location.filename,int(op.location.start_line)
        except (ValueError,AttributeError): parsed_file,parsed_line=None,None
        if self.text_locations and parsed_file=='-':
            candidate=parsed_line
            if 1<=candidate<=len(self.lines) and op.name in self.lines[candidate-1]: line=candidate
        return {'node_path':self.paths.get(op,'module'),'ir_location':loc,
                'stage_text_line':line,'line_mapping':'parsed_text_checked' if line else 'unavailable'}

    def kind(self,type_):
        # Type strings select scalar dtypes only; no operation/region text parsing.
        kind=str(type_)
        if kind=='i1': return 'bool'
        dtype(kind)
        return kind

    def memtype(self,type_):
        try: spec=self.ir.MemRefType(type_)
        except ValueError as exc: raise Unsupported('仅支持 ranked memref，tensor/vector 尚未支持') from exc
        if spec.memory_space is not None and str(spec.memory_space) not in {'0','0 : i64','0 : i32'}:
            raise Unsupported('非默认 memref memory_space 尚未支持')
        self.kind(spec.element_type)
        return spec

    def layout(self,spec,buf):
        strides,offset=spec.get_strides_and_offset()
        dynamic=self.ir.ShapedType.get_dynamic_stride_or_offset()
        actual=tuple(v//buf.dtype.itemsize for v in buf.values.strides)
        if offset not in (0,dynamic): raise Unsupported('非零参数/局部 offset 需显式 view 契约')
        if len(strides)!=len(actual) or any(s!=dynamic and s!=v for s,v in zip(strides,actual)):
            raise Unsupported('memref 声明 stride 与实际存储不匹配')

    def value(self,value,env):
        if value not in env: raise Unsupported('官方 SSA 值未绑定')
        return env[value]

    def sequence(self,block,env):
        for view in block.operations:
            op=view.operation;self.budget.tick()
            location=self.location(op);line=location['stage_text_line'] or 0
            try:
                with self.budget.at('mlir_operation',location['stage_text_line'],node_path=location['node_path'],operation=op.name):
                    result=self.operation(op,env,line)
            except SemanticFault as exc:
                if exc.line==0:
                    exc.line=None;exc.args=(str(exc).removeprefix('L0: '),)
                exc.details.setdefault('mlir_location',location)
                raise
            except Unsupported as exc:
                if not getattr(exc,'mlir_location',None): exc.mlir_location=location
                raise
            if result is not None: return result
        return None

    def write(self,buf,index,value,op,line):
        if self.pure_only: raise Unsupported('数据相关符号分支含内存写；仅执行具体输入')
        buf.write(index,value,line)
        if buf.position is not None: self.writer_locations[(buf.position,tuple(map(int,index)))]=self.location(op)

    def operation(self,op,env,line):
        name=op.name
        operands=[self.value(a,env) for a in op.operands]
        if name in {'scf.yield','func.return','linalg.yield','scf.reduce'}: return (name,tuple(operands))
        if name=='scf.for':
            lo,hi,step=map(int,operands[:3]);carried=tuple(operands[3:])
            if step<=0: raise Unsupported('scf.for 需要正步长')
            block=op.regions[0].blocks[0]
            for i in range(lo,hi,step):
                local=dict(env);local.update(zip(block.arguments,(np.int64(i),*carried)))
                yielded=self.sequence(block,local)
                if yielded is None or yielded[0]!='scf.yield': raise Unsupported('scf.for 缺少 yield')
                carried=yielded[1]
            values=carried
        elif name=='scf.parallel':
            block=op.regions[0].blocks[0];n=len(block.arguments)
            lo,hi,step=operands[:n],operands[n:2*n],operands[2*n:]
            if any(int(s)<=0 for s in step): raise Unsupported('scf.parallel 需要正步长')
            region=self.budget.new_region()
            for index in itertools.product(*[range(int(a),int(b),int(s)) for a,b,s in zip(lo,hi,step)]):
                local=dict(env);local.update(zip(block.arguments,map(np.int64,index)))
                with self.budget.iteration(region,index): self.sequence(block,local)
            return None
        elif name=='scf.if':
            cond=operands[0]
            def branch(i):
                r=op.regions[i]
                if not len(r.blocks): return ()
                yielded=self.sequence(r.blocks[0],dict(env))
                if yielded is None or yielded[0]!='scf.yield': raise Unsupported('scf.if 缺少 yield')
                return yielded[1]
            if isinstance(cond,Expr):
                if not len(op.results) or not len(op.regions[1].blocks): raise Unsupported('数据相关符号 if 需要有值的纯分支')
                old=self.pure_only;self.pure_only=True
                try: yes,no=branch(0),branch(1)
                finally: self.pure_only=old
                values=tuple(select(cond,a,b) for a,b in zip(yes,no))
            else: values=branch(0 if bool(cond) else 1)
        elif name=='memref.store':
            self.write(operands[1],tuple(operands[2:]),operands[0],op,line);return None
        elif name=='memref.copy':
            source,target=operands
            if source.shape!=target.shape or source.dtype!=target.dtype or np.shares_memory(source.values,target.values):
                raise Unsupported('memref.copy shape/dtype 不同或源目标重叠')
            for idx in np.ndindex(source.shape): self.write(target,idx,source.read(idx,line),op,line)
            return None
        elif name=='linalg.fill':
            if len(operands)!=2 or len(op.results): raise Unsupported('仅支持 buffer 形式 linalg.fill')
            block=op.regions[0].blocks[0];body=list(block.operations)
            if len(body)!=1 or body[0].operation.name!='linalg.yield' or list(body[0].operation.operands)!=[block.arguments[0]]:
                raise Unsupported('linalg.fill 的 region 不符合标准标量填充')
            value,buf=operands
            for idx in np.ndindex(buf.shape): self.write(buf,idx,value,op,line)
            return None
        elif name in {'memref.alloc','memref.alloca'}:
            if self.pure_only: raise Unsupported('数据相关符号分支含 allocation')
            spec=self.memtype(op.results[0].type);supplied=iter(operands)
            shape=tuple(int(next(supplied)) if n<0 else int(n) for n in spec.shape)
            if next(supplied,None) is not None: raise Unsupported('allocation affine symbols 尚未支持')
            self.budget.shape(shape)
            buf=Buffer(np.empty(shape,dtype=dtype(self.kind(spec.element_type))),initialized=False,
                symbolic=self.symbolic,budget=self.budget,name=self.paths[op],allocation_line=line or None)
            self.layout(spec,buf);values=(buf,)
        elif name=='memref.load':
            values=(cast(operands[0].read(tuple(operands[1:]),line),self.kind(op.results[0].type)),)
        elif name=='memref.dim':
            axis=int(operands[1])
            if not 0<=axis<len(operands[0].shape): raise SemanticFault('memref.dim 轴越界',rule_id='MEM-CONTRACT',line=line or None)
            values=(np.int64(operands[0].shape[axis]),)
        elif name=='memref.subview':
            base=operands[0];dynamic=iter(operands[1:]);sentinel=self.ir.ShapedType.get_dynamic_stride_or_offset()
            arrays=[tuple(int(v) for v in op.attributes[k]) for k in ('static_offsets','static_sizes','static_strides')]
            offsets,sizes,strides=[tuple(int(next(dynamic)) if v==sentinel else v for v in a) for a in arrays]
            if next(dynamic,None) is not None: raise Unsupported('subview 动态参数不匹配')
            spec=self.memtype(op.results[0].type);dims=spec.shape;choices=[]
            for drop in itertools.combinations([i for i,n in enumerate(sizes) if n==1],len(sizes)-len(dims)):
                kept=[n for i,n in enumerate(sizes) if i not in drop]
                if all(d<0 or int(d)==n for d,n in zip(dims,kept)): choices.append(drop)
            if len(choices)!=1: raise Unsupported('subview 降维映射不唯一')
            buf=base.subview(offsets,sizes,strides,choices[0])
            if storage_type(buf.dtype)!=storage_type(dtype(self.kind(spec.element_type))): raise Unsupported('subview 不能改变 dtype')
            declared,offset=spec.get_strides_and_offset();actual=tuple(v//buf.dtype.itemsize for v in buf.values.strides)
            root=base
            while hasattr(root,'_parent'): root=root._parent
            actual_offset=(buf.values.ctypes.data-root.values.ctypes.data)//buf.dtype.itemsize
            if offset not in (sentinel,actual_offset) or any(s!=sentinel and s!=v for s,v in zip(declared,actual)):
                raise Unsupported('subview 声明 offset/stride 与共享存储不匹配')
            values=(buf,)
        elif name=='arith.constant':
            attr=op.attributes['value'];kind=self.kind(op.results[0].type)
            if isinstance(attr,(self.ir.IntegerAttr,self.ir.BoolAttr)): value=attr.value
            elif isinstance(attr,self.ir.FloatAttr):
                value=attr.value
                if np.isnan(value): raise Unsupported('FloatAttr 的 NaN payload 尚未核实；不通过宿主 float 近似')
            else: raise Unsupported('仅支持标量 IntegerAttr/FloatAttr 常量')
            values=(cast(value,kind),)
        elif name in self.binary:
            kind=self.kind(op.results[0].type)
            if name in {'arith.andi','arith.ori'} and kind!='bool': raise Unsupported('当前 andi/ori 只支持 i1')
            values=(apply(self.binary[name],operands,kind),)
        elif name in self.unary: values=(apply(self.unary[name],operands,self.kind(op.results[0].type)),)
        elif name in self.casts:
            source,target=self.kind(op.operands[0].type),self.kind(op.results[0].type)
            values=(mlir_cast(name.split('.')[1],operands[0],source,target),)
        elif name=='arith.cmpi':
            predicate=int(op.attributes['predicate'].value)
            predicates=('eq','ne','slt','sle','sgt','sge','ult','ule','ugt','uge')
            values=(mlir_cmpi(predicates[predicate],*operands,self.kind(op.operands[0].type)),)
        elif name=='arith.cmpf':
            predicates=('false','oeq','ogt','oge','olt','ole','one','ord','ueq','ugt','uge','ult','ule','une','uno','true')
            pred=predicates[int(op.attributes['predicate'].value)];a,b=operands
            if isinstance(a,Expr) or isinstance(b,Expr):
                nodes=(literal(a),literal(b));value=Expr('cmpf_'+pred,nodes,'i1',merge_trace(nodes))
            else:
                unordered=bool(np.isnan(a) or np.isnan(b))
                special={'true':True,'false':False,'ord':not unordered,'uno':unordered}
                if pred in special: value=np.bool_(special[pred])
                else:
                    test=bool(apply({'eq':'eq','ne':'ne','lt':'lt','le':'le','gt':'gt','ge':'ge'}[pred[1:]],[a,b]))
                    value=np.bool_(unordered or test if pred[0]=='u' else not unordered and test)
            values=(value,)
        elif name=='arith.select': values=(select(*operands),)
        else: raise Unsupported(f'未支持官方 MLIR 操作 {name}')
        if len(values)!=len(op.results): raise Unsupported(f'{name} 的结果数量不匹配')
        for key,value in zip(op.results,values): env[key]=trace_value(value,self.paths[op],line,name)
        return None

    def run(self,buffers,symbols=None,symbolic=False):
        self.budget=buffers[0].budget if buffers else Budget(self.max_steps)
        self.symbolic,self.pure_only,self.writer_locations=symbolic,False,{}
        with self.context:
            block=self.function.regions[0].blocks[0]
            if len(block.arguments)!=len(buffers): raise Unsupported('MLIR 参数数量与用例不同')
            env={}
            for arg,buf in zip(block.arguments,buffers):
                spec=self.memtype(arg.type)
                if len(spec.shape)!=len(buf.shape) or any(d>=0 and d!=n for d,n in zip(spec.shape,buf.shape)):
                    raise Unsupported('MLIR 参数 shape 与用例不同')
                if storage_type(dtype(self.kind(spec.element_type)))!=storage_type(buf.dtype):
                    raise Unsupported('MLIR 参数 dtype 与用例不同')
                self.layout(spec,buf);env[arg]=buf
            returned=self.sequence(block,env)
            if returned is None or returned[0]!='func.return' or returned[1]:
                raise Unsupported('目前只支持通过 memref 输出的 void 函数')
        return {'steps':self.budget.steps,'backend':'mlir_operation_objects',
                'execution_model':'structured logical loops; serial enumeration of parallel iterations',
                'location_model':'official parsed locations when checked; otherwise operation paths'}
