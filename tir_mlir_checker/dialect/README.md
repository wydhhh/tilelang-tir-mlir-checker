# `tir_mlir_checker.dialect` — Dialect 层契约检查

本子包是课题新增的**方言层**，与宿主 `tir_mlir_checker` 的 IR 级检查相互独立：

```text
IR 级（宿主）              方言层（本子包）
完整性/事实/规则/解释器     方言索引 + 目标契约（profile）核验
回答“数值有没有变”         回答“交付形态是不是契约要求的东西”
```

## 快速使用

```bash
# 盘点模块里出现哪些方言/操作/属性/类型
tir-mlir-dialect inventory --module 03_mlir_linalg.mlir

# 按契约检查一个模块
tir-mlir-dialect check --module 03_mlir_linalg.mlir --profile ame-candidate

# 直接检查 Trace 的 Stage 03（指纹由宿主 trace_io 校验）
tir-mlir-dialect check --trace <算子目录> --stage 03 --profile ame-candidate --out /tmp/dialect-001

# 厂商文档到位后：从现状生成契约骨架，填数据即可
tir-mlir-dialect template --module with_ame.mlir --namespace ame > ame.json
```

`python -m tir_mlir_checker.dialect ...` 与上面的命令等价。

## 状态语义（重要）

| 条款状态 | 违反时报告 | 含义 |
| --- | --- | --- |
| `confirmed` | `mismatch`（exit 1） | 已确认的契约，违反即错误 |
| `candidate` | `uncertain` | 候选契约，违反只标记待复核 |
| `pending` | `uncertain` | 未确认，既不通过也不报错 |
| 未登记方言 | `unsupported`（exit 4） | 无法判断，绝不静默通过 |

`DIA-PROFILE-PENDING` 汇总所有未确认条款；它出现在结果里是**正常**的，前提是 profile 尚未完成。

## 扩展方式

- 新方言：写 profile JSON（`dialects.<ns>.ops`），不需要改代码；
- 新平台：另写 profile；
- 新检查维度：在 `analyzers.py` 加一个纯函数并在 `analyze_module` 注册；
- 新读取后端：实现 `read_module` 兼容函数。

## 后端强度

| | official（`mlir` bindings） | text（纯标准库） |
| --- | --- | --- |
| 操作名 | 规范名，含隐式 region/terminator | 打印视图；`return` 等 elided 写法经别名规范化 |
| 操作数/结果/region 计数 | 精确（arity 检查只在此后端启用） | 同行近似，不用于 arity |
| 属性 | 官方属性对象 | 同行平衡 `{...}` 的键值 |
| 位置 | `op.location` | 文本行号 |
| verifier | 可运行 | 不可 |

`--backend auto` 优先 official，失败时回退 text，并**把回退原因写进结果**（`module.fallback_reason`）；强制 `official`/`text` 时不会静默切换。

## 文件地图

| 文件 | 职责 |
| --- | --- |
| `model.py` | 事实与契约数据类、错误类型 |
| `profiles.py` | 内置 `generic` 与 `ame-candidate` |
| `profile.py` | profile 校验/加载/序列化/模板 |
| `text_reader.py` / `official_reader.py` | 两种读取后端 |
| `reader.py` | 后端选择与统一入口 |
| `verify.py` | 官方 verifier 桥接 |
| `analyzers.py` | `DIA-*` 规则 |
| `report.py` | 兼容宿主诊断 schema 的 JSON + 三段式 md |
| `cli.py` | `tir-mlir-dialect` |
