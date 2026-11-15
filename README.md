# TileLang TIR→MLIR Checker

用于检查 TileLang 的 TIR 到结构化 MLIR 转换，提供五阶段 IR 导出、完整性校验、基础规则检查、受限模拟执行、诊断和证据。Agent 可通过命令调用工具，回查原始 IR 并生成语义分析报告。

## 安装与一个完整示例

Python 3.10+。静态检查只依赖标准库；模拟执行另需 NumPy 和 ml_dtypes。包不附带 TileLang、TVM、MLIR 或服务器配置。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[execution]'
tir-mlir-check --version

# 使用仓库内的真实 Where IR 和独立 NumPy 参考。
# 输出必须是新目录；重复运行请换名。
tir-mlir-check check --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --operator Where --dynamic --case tests/adapters/where_case.py \
  --out /tmp/where-check-001

tir-mlir-check show --trace tests/fixtures/traces/十算子-2026-11-11/Where \
  --stage 03 --start 60 --end 85
```

示例会分别模拟五份 IR，并与同一组独立参考对照。模拟覆盖本包支持的 IR 子集；结果中会注明缺少的依赖和未支持的操作。样本对照与机器码执行、任意输入等价分别记录。

### 结果示例：Where

上面的命令检查 `out = np.where(cond != 0, x, y)`。输入为 `cond: uint8[2048]`、`x/y: f32[2048]`；9 组用例覆盖全零、非零、交替、随机、块边界及 NaN/Inf/正负零，独立参考来自 NumPy。

实际 stdout JSON 摘录：

```json
{
  "tool_status": "ok",
  "dynamic_status": "passed",
  "semantic_conclusion": "bounded_checks_passed",
  "summary": {"match": 20, "uncertain": 4, "mismatch": 0}
}
```

| 结果 | 本例内容 |
| --- | --- |
| 五阶段数值对照 | 9 组 × 5 阶段，45/45 次逐位一致，最大有限绝对误差 0 |
| 自动规则汇总 | 20 项 match、4 项 uncertain、0 项 mismatch；这是检查条目数，与执行次数分开统计 |
| Agent 复核 | 4 项 uncertain 是公式、数据流、精度和控制流的复核清单，Agent 结合原始 IR 判断 |
| 执行方式 | 本包轻量解释器；本例未调用 TVM/MLIR 官方验证或执行机器码 |

命令在 `/tmp/where-check-001/` 生成 `Where 自动检查记录.md` 和 `Where 检查证据.zip`。Agent 读取记录与证据，用 `show` 回查原始 IR 后，另写 `Where 单算子完整检查报告.md`。

**发现错误时会返回什么？** 在同一 Trace 的临时副本中，将 Stage 03 第 167 行 `arith.cmpi ne` 改成 `eq`，刷新副本 manifest 后，用相同用例检查：退出码为 **1**，Stage 03 的 9 组均为 mismatch，其余阶段的 36 次对照通过。证据包的 `diagnostics.txt` 给出如下诊断（省略目录前缀）：

```text
03_mlir_linalg.mlir:226: error: 实际输出与独立参考不一致 [NUM-OUTPUT-DIFFERENCE; mismatch]
```

首个反例是 `zero` 用例的 `out[0]`：实际为 `0.25`，参考为 `-4096.5`。第 226 行是结果写回位置；Agent 顺着写回、临时区和条件分支回查到第 167 行，确认比较条件反转。工具提供反例与定位线索，Agent 解释具体原因。

## 四个命令与六类核心功能

| 命令 | 用途 |
| --- | --- |
| `export --adapter ... --out ...` | 在兼容的 TileLang RISC-V 环境中真实编译，导出一个算子的五阶段 |
| `run --adapter ... --operator ... --out ...` | `export → check` 的薄封装，默认包含动态对照；`--static-only` 仅静态检查 |
| `check --trace ...` | 检查已有 Trace；`--dynamic --case ...` 增加模拟和独立参考对照 |
| `show --trace ... --stage 03 --start 60 --end 85` | 校验输入后返回原始 IR 片段、指纹和行号 |

命令不在 PATH 时可用 `python -m tir_mlir_checker`。Python API 是 `tir_mlir_checker.check()`，参数以 CLI 帮助和源码签名为准。

1. **完整性校验**：核对五阶段、manifest、字节数、行数及 SHA-256；哈希只能绑定文件，不能独立证明它们来自同一次编译。
2. **IR 事实提取**：提取参数、形状、类型、循环、读写和转换，保留阶段与位置；轻量解析与官方解析分别标明。
3. **基础错误检查**：在支持的模型和显式契约下检查越界、未初始化读取、漏写、只读输入改写及写冲突。
4. **有限动态对照**：独立模拟各阶段，用显式输入与独立参考比较；保留位模式/容差、NaN 和正负零策略。
5. **来源与定位**：保存触发条件、首个反例、相关读写及输入→临时区→输出导航；导航最多 64 条，可能截断，不是自动根因证明。
6. **稳定诊断与证据**：JSON 保留 `rule_id/status/method/scope/location/note`，证据包内有 MLIR 风格 `diagnostics.txt`。

转换次数或表达式外观不同不独立构成错误。规则不按算子名称内置数学公式；未知操作、缺参考、未执行、超时和具体反例分开报告。

## 输入与官方前端

五阶段是 `00_tir_initial`、`01_tir_legalized`、`02_tir_optimized`、`03_mlir_linalg`、`04_mlir_final`。主检查边界为 **02→03**；04 仍在 LLVM 转换之前。01/02 内容相同也保留，这是选定检查点，不是每个 pass 的记录。

已有 Trace 传入完整单算子目录。`export` 同时保存 00–02 的 `*_object.json`；manifest 绑定对象、阶段文本和 TVM 版本/commit。有 MLIR Python 绑定时还记录依赖身份。传递文件时保留整个目录，包括对象和 manifest。

| 路线 | 解析/验证 | 模拟执行与限制 |
| --- | --- | --- |
| 默认 `ast` / `text` | Python AST 与有限 MLIR 文本解析 | 本包轻量解释器；没有运行官方 verifier |
| `--tir-backend tvm` | 官方 `ir.load_json`；无绑定对象时尝试 TVMScript parser；检查 `verify_ssa` / `verify_well_formed` | 本包直接访问 TIR 对象节点；结果在 `tir_verification` |
| `--mlir-backend official` | 注册方言下的 `Module.parse` / `operation.verify()` | 本包直接访问 SSA、属性和 regions；结果在 `mlir_python_verification` |
| `--mlir-opt /path/to/mlir-opt` | 独立验证 03/04，不附加转换 passes | 只验证结构，结果在 `mlir_verification` |

有匹配的官方依赖时可以组合使用：

```bash
tir-mlir-check check --trace /path/to/trace --operator Where \
  --tir-backend tvm --mlir-backend official \
  --ir-python /path/to/compatible/python \
  --dynamic --case /path/to/adapter.py --out /tmp/where-official-001
```

`--ir-python` 指向同机兼容 Python；`--tir-python` 可用于 TIR 官方前端，两者同时设置时须相同。运行环境由调用者配置，服务器连接和工具链安装由 Agent 处理。显式选择官方路线后，解析或依赖不兼容会返回诊断，不回退到轻量解析；MLIR 要求操作所属方言已注册。

官方结构验证和本包模拟分别记录。绑定对象会核对其打印文本与原阶段；版本、扩展、方言或属性不兼容保留诊断。MLIR 依赖身份按 manifest 核对，未提供时注明。TIR `original_span` 和 MLIR 显式 `loc` 可能指向 DSL 源码，无法对应 Trace 文本行时保留节点路径与原位置。

执行范围随前端区分，以 `support_inventory` 和实际诊断为准。支持的循环、内存、标量运算与转换保留原位宽和次序；f32↔bf16/f16 默认 RNE，算术 NaN 按显式分类契约比较，不承诺 payload。TVM 对象支持标准未调度 Block init（完整直接轴映射、串行归约），MLIR 对象支持受限 `scf.for iter_args`。调度/rfactor、并行归约、复杂布局、未知属性/fastmath、物理调度及特殊同步不作通用支持承诺。逻辑并行模拟不能证明真机线程映射。

## Agent 使用流程

1. **准备输入**：取得完整 Trace；若需导出，Agent 在已授权服务器配置兼容环境和适配。适配的 `make_kernel()` 返回未编译 PrimFunc，`compile_options` 显式指定 target/out_idx，模块顶层不要编译。动态用例由 `make_cases(seed)` 返回 [TestCase](tir_mlir_checker/cases.py)，明确参数、输出、可写 inouts、独立参考和比较策略。参考不能来自待测阶段；示例见 [Where 适配](tests/adapters/where_case.py)。服务器连接、上传、下载及凭证由 Agent 管理，凭证不写入源码或日志。
2. **调用工具**：在兼容环境用 `export/run` 真实采集，或用 `check` 读取现有目录；逐次使用新输出路径。读取退出码、stdout JSON 的 `tool_status/dynamic_status/semantic_conclusion/ir/agent_diagnostics/artifacts`，再读证据中的完整结果、事实与诊断。采集失败可能只留部分 IR，不能称五阶段完成。
3. **回查与判断**：先看具体 error，再看 uncertain/unsupported。用 `show` 优先核对 02→03 的参数/布局、公式与初始化、源头下标、精度/归约顺序、守卫、写回覆盖及同步。区分报错表现位置与根因位置。契约不匹配、缺参考、超时或不支持不自动成为算子缺陷，不改写自动标签。
4. **补充与报告**：明确的解释缺口可用下方翻译接口补执行；真机测试另配环境并保存证据。官方验证、内置模拟、Agent 翻译和真机结果分别记录。报告内容见下表。

case、TVMScript parser 和翻译适配会处理普通 Python；子进程与超时不是安全沙箱，应使用可信输入。不得为取得通过而放宽容差、删除未知操作或省略同步。

| 报告部分 | 内容 |
| --- | --- |
| 检查结论 | 执行状态、已测结果与具体错误 |
| 测试范围与结果 | 算子/版本/批次、shape/dtype、输入组数、参考与比较策略；各类验证分别列结果 |
| 问题与 Agent 判断 | 问题（原状态/编号）｜关键位置/反例｜Agent 判断及前提；没有问题就直说 |

调用时用 `--operator <算子名称>`；省略则按 manifest.name、唯一 IR 函数名、Trace 目录名依次取名。展示名称与各阶段真实函数名同时保留，不用于选择数学契约。名称可含字母、数字、下划线、连字符和点，不能含空白或路径。建议输出目录为 `<算子名称>/<日期与运行ID>/`。

| 产物 | 用途 |
| --- | --- |
| stdout JSON | 自动状态、IR 索引、诊断和产物路径 |
| `<算子名称> 自动检查记录.md` | Python 自动记录，保留原标签 |
| `<算子名称> 检查证据.zip` | 完整诊断、事实、数组与执行证据 |
| `<算子名称> 单算子完整检查报告.md` | Agent 另写；一级标题与文件名文字一致 |

以 `artifacts` 返回的真实位置为准；`final_report` 是命名提示，不代表已生成最终报告。长日志、完整命令、数组和规则明细放证据，报告附关键行号与证据链接。

退出码：**0** 请求完成，**1** 原检查具体 mismatch，**2** 输入错误，**3** 工具错误/超时，**4** 不支持。退出 0 或静态 ok 不能替代动态结果；有限样本通过不能推广为任意输入、机器码或 AME 通过。

<details>
<summary>Agent 翻译补执行协议（解释器不支持时）</summary>

先区分缺参考、预算不足与语义缺口。缺参考先补独立契约；超时先缩小样本。语义明确时分别按原 IR 翻译阶段模型，优先 02 和 03，不从上层公式重写两个相同答案。

```bash
tir-mlir-check check --trace /path/to/trace --operator abs \
  --dynamic --case /path/to/case.py --translation /path/to/translation.py \
  --out /tmp/abs-translation-001
```

接口示意（哈希、行号和函数体须来自实际 IR）：

```python
import numpy as np

def run_stage(arguments, symbols):
    x, _ = arguments
    y = np.empty_like(x)
    for i in range(x.size):
        y[i] = np.abs(x[i])
    return {1: y}

TRANSLATIONS = {
    '03_mlir_linalg': {
        'sha256': '<原始阶段文本的 SHA-256>',
        'entry': 'run_stage',
        'line_map': [{'ir': [1, 10], 'python': [4, 10]}],
        'assumptions': ['固定 f32 连续一维输入，无别名；逻辑串行枚举']
    }
}
```

- 函数只接收参数副本与动态符号，不接收参考答案，返回 `{输出参数位置: ndarray}`；覆盖全部输出，保留 inouts 初值。保留原搬运、下标、初始化、dtype、舍入和归约次序，不能默默省略 extern/barrier/shuffle。
- `sha256` 绑定原 IR；`line_map` 是原 IR 与 Python 的闭区间映射。工具只验证格式、范围与指纹，翻译忠实性由 Agent 复核。NumPy 默认广播、负下标、溢出和归约可能与 IR 不同，必须显式处理。
- 补充结果在 `agent_translation` 单列；`input_binding=identical` 才称同输入补充验证。原输入不可用时会明确标记，不能假设绑定。
- 原 unsupported/mismatch、主结论及退出码不被补充结果升级。生成代码报错先查翻译；通过也只称 Agent 翻译后的样本验证，不继承内置解释器的内存/来源/冲突证明。
- 翻译受子进程超时约束，不受解释器 max_steps 计数。证据中 `agent_translation/` 保存代码、适配、阶段快照和数组；数组按原始 uint8 与 array_dtypes 恢复。包不内置模型或自动翻译器。

</details>
