# Agent error log

- 2026-07-13｜并行环境探测被一个失败命令整体中断｜未隔离预期可能失败的 Git/Conda 子命令｜改为逐项捕获结果并在正确仓库根目录运行｜后续并行探测对每个命令单独处理退出码
- 2026-07-13｜复制根 AGENTS.md 时调用了 V8 环境不存在的 `atob` 和 `TextDecoder`｜错误假设 JavaScript Web API 可用｜改用显式 Base64 与 UTF-8 解码后通过 apply_patch 写入｜使用运行时已声明能力或先做无副作用能力检查
- 2026-07-13｜`conda run` 转发 pytest 输出时发生 GBK 编码异常｜未预设 Windows 中文路径下的 UTF-8 输出｜中间测试直接调用 DeepWater 解释器，最终命令设置 UTF-8 并验证｜Windows 自动化命令显式设置 `PYTHONUTF8=1`
- 2026-07-13｜editable 安装生成的 UTF-8 `.pth` 使 Python 3.10 无法在 GBK 下启动｜仓库中文路径被写入 site-packages 路径文件｜只删除本次生成的 RiverLagNet `.pth`，改用普通安装｜非 ASCII 路径下不使用 editable 安装
- 2026-07-13｜pytest warning 正则令 pyproject.toml 解析失败｜TOML basic string 中反斜杠未转义｜改用 TOML literal string并重跑测试｜正则配置优先使用 literal string
- 2026-07-13｜RuntimeStatsCallback 在 `on_fit_end` 调用 `self.log` 导致成功训练被判失败｜未遵循 Lightning 对回调生命周期的日志限制｜增加 fast-dev 回归测试并改为直接调用 Trainer loggers｜所有自定义回调必须纳入真实 Trainer 生命周期测试
- 2026-07-13｜普通安装后 CLI 找不到 Hydra 根配置｜源码相对路径没有考虑 wheel 内模块位置｜增加 package-data 配置镜像及一致性测试，CLI 改用包内路径｜发布入口必须在非 editable 安装后执行 smoke test
- 2026-07-13｜GPU `16-mixed` 下 attention 切片赋值发生 FP16/FP32 冲突｜CPU 测试未覆盖 softmax 在 autocast 中提升精度的行为｜增加 CUDA autocast 测试，attention 固定 FP32 并在消息乘法边界显式转换｜混合精度模型必须包含真实 CUDA 前向回归测试
- 2026-07-13｜输入变量多于三个时目标标准化维度不匹配｜先裁目标通道再应用全特征 scaler｜增加额外动态变量测试并改为先标准化全特征再取前三个目标｜数据测试必须覆盖 `V>target_dim`
