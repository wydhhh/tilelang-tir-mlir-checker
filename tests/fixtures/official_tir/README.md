# 官方 TIR 前端验收输入

where与sum各含真实00–04文本、00–02的同次TVM对象及精简manifest。来源为2026-11-13已采集的十算子改错实验记录，原文件由`/tmp/checker-survey-20261113/ten/traces/`的采集证据提取后逐字节保留；完整历史证据在本机archive的对应验证归档，不作为测试运行依赖。

- Where：长度35、tile32；独立参考包括全零、非零与混合条件，按位比较。
- Sum：输入[2,33]、归约tile256；独立NumPy参考包含random、positive、ramp，显式容差。
- TVM：0.23.dev0，commit `882a774844993d103ae6e317ba3c7bbb5952b662`；TileLang项目commit `57c548cc8d6b3f581d6687427c44183faaded981`。

manifest分别绑定阶段文本与对象SHA-256；本轮使用相同项目环境重新载入、检查、模拟，不把原历史结果当新执行。小型case文件不导入算子源代码，不按算子名称分派引擎。版本不匹配须返回unsupported，不为让测试通过而修改对象或跳过验证。
