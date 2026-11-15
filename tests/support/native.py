"""Optional native MLIR execution in a disposable worker; no SSH or installation."""
import hashlib
import os
import platform
from pathlib import Path

import numpy as np

from tir_mlir_checker.execution import Unsupported


class NativeMLIR:
    def __init__(self, source, symbol, directory):
        if platform.machine() != 'riscv64':
            raise Unsupported('该项目原生后端需要 riscv64 与现有 TileLang/MLIR 工具链')
        if os.environ.get('LD_PRELOAD'):
            raise Unsupported('原生执行进程必须在启动前排除 LD_PRELOAD')
        maps=Path('/proc/self/maps')
        if maps.exists() and 'libbf16shim' in maps.read_text():
            raise Unsupported('原生进程加载了旧 bf16 shim')
        try:
            from tilelang.jit.adapter.riscv.wrapper import load_host_module
        except ImportError as exc:
            raise Unsupported(f'现有 TileLang 原生执行依赖不可用：{exc}') from exc
        self.module=load_host_module(source,function_name=symbol,path=Path(directory)/'kernel.so')
        self.info={'backend':'tilelang_host_mlir','machine':platform.machine(),
                   'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                   'library_sha256':hashlib.sha256(Path(self.module.path).read_bytes()).hexdigest(),
                   'llvm_root':os.environ.get('TILELANG_RISCV_LLVM_ROOT'),
                   'ld_preload_absent':not bool(os.environ.get('LD_PRELOAD')),
                   'note':'此后端执行保存的 MLIR，再经 LLVM/运行库；没有重新执行 TIR→MLIR。'}

    def run(self,case):
        arrays=[a.copy() for a in case.arguments]
        for pos in case.outputs:
            if pos in case.inouts:continue
            arrays[pos].view(np.uint8).reshape(-1)[:]=0xa5
        self.module(*arrays)
        for pos,a in enumerate(arrays):
            if pos not in case.outputs and pos not in case.inouts and not np.array_equal(a.view(np.uint8),case.arguments[pos].view(np.uint8)):
                raise ValueError(f'原生代码修改了只读输入参数 {pos}')
        return [arrays[pos] for pos in case.outputs]

    def close(self):
        self.module.close()
