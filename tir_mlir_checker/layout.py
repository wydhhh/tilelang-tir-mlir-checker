"""Decode a bounded TVM Fragment JSON graph without importing TVM or executing it."""
import json
from .execution import Unsupported


class Fragment:
    def __init__(self, payload, index):
        graph=json.loads(payload) if isinstance(payload,str) else payload
        self.nodes=graph['nodes']
        if len(self.nodes)>10000:raise Unsupported('Fragment metadata 超过节点上限')
        root=self.nodes[graph['root_index']]
        if root['type']!='ffi.Map':raise Unsupported('metadata 根不是 Map')
        pairs=root['data'];found=None
        for k,v in zip(pairs[::2],pairs[1::2]):
            if self.nodes[k].get('data')=='tl.Fragment':found=self.nodes[v]['data'][index]
        if found is None:raise Unsupported('缺少 tl.Fragment metadata')
        node=self.nodes[found]
        if node['type']!='tl.Fragment':raise Unsupported('metadata 类型不匹配')
        self.fragment=node['data']
        self.sizes=[self.evaluate(n,{}) for n in self.nodes[self.fragment['input_size']]['data']]
        if len(self.sizes)!=2 or self.evaluate(self.fragment['replicate_size'],{})!=1:
            raise Unsupported('当前仅支持 rank2、replicate_size=1 的完整 Fragment')
        self.evaluate(self.fragment['forward_thread'],{'_i':0,'_j':0})

    def evaluate(self,index,variables,depth=0):
        if depth>64:raise Unsupported('Fragment 表达式嵌套过深')
        node=self.nodes[index];kind=node['type'];data=node.get('data',{})
        if kind=='ir.IntImm':return int(data['value'])
        if kind=='tir.Var':
            key=self.nodes[data['name']]['data']
            if key not in variables:raise Unsupported(f'未支持的 Fragment 变量 {key}')
            return variables[key]
        operations={'tir.Add':lambda a,b:a+b,'tir.Sub':lambda a,b:a-b,'tir.Mul':lambda a,b:a*b,
                    'tir.FloorDiv':lambda a,b:a//b,'tir.FloorMod':lambda a,b:a%b}
        if kind not in operations:raise Unsupported(f'未支持的 Fragment 节点 {kind}')
        a,b=[self.evaluate(data[k],variables,depth+1) for k in ('a','b')]
        if kind in {'tir.FloorDiv','tir.FloorMod'} and b<=0:raise Unsupported('Fragment 非正除数')
        return operations[kind](a,b)

    def owner(self,i,j):
        return self.evaluate(self.fragment['forward_thread'],{'_i':i,'_j':j})
