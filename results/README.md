# 实验结果

- `main/`：正式的首轮 primitive 对照，单线程、每项三次在线重复，包含 CKKS
  矩阵实验和 FHEW 标量联合查表。
- `smoke/`：构建后的一次重复小规模正确性检查，不用于正式性能结论。
- `stage_a/`：真实 checkpoint 的跨 expert 低秩结构分析及校准输出误差。
- `stage_b/`：秘密多位 ID/标量 activation 的 exact 有限域实验、原始数据与资源边界。
- `stage_c/`：同一秘密输入的 TFHE-rs many-LUT 多输出复用、容量门槛及未测对照。
- `stage_d/`：`N=2` 矩形 tiled 矩阵的有限域原型、原始数据与扩展边界。
- `stage_e/` 和 `stage_f/`：前置验收不足，记录未运行的条件与后续缺口。

每个阶段目录包含可读报告；运行过的阶段保留相应原始结果和实验配置。
`main/` 和 `smoke/` 使用合成权重；`stage_a/` 分析真实 MoE checkpoint；
`stage_b/`、`stage_c/`、`stage_d/` 使用有限定点域的公开小规模权重。
它们都不代表端到端 MoE 模型性能。
