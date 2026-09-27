# FHEW Select-and-Apply candidate

目标是不依赖跨-expert 低秩结构，让小的 encrypted expert ID 控制大的 plaintext linear
operator，同时不生成完整 `Enc(W_e)`，也不退化成 one-hot Dense-FHE。

`fused_scalar_lut.cpp` 是历史的 1-bit ID、1-bit activation 基线。阶段 B 新增
`stage_b_scalar.cpp`，在 STD128 下比较 multi-bit 联合 LUT、负循环扩展和逐位 Boolean
CMUX tree，并允许传入 [-2,2] 范围内的任意公开权重表。TFHE-rs 的独立 WOPBS-only
原型位于 `experiments/tfhe_stage_b/`；其 legacy 参数尚未完成与 OpenFHE 的安全级别
对齐。实验方法与实测边界见 `results/stage_b/report.md`。仍未解决：

1. 超出本阶段窄定点域的多精度 activation；
2. packed vector、多输出复用以及 vertical packing 的同口径验证；
3. tiled / packed matrix operator；
4. CKKS packed arithmetic 与 FHEW/TFHE encrypted control 的转换和布局；
5. 对任意 plaintext expert weights 的完整 `Enc(W_e x)`。

每实现一个缺口前，必须先检索 OpenFHE 及其他开源 FHE 项目中可复用的 PBS、scheme
switching、LUT packing、CMux 或 private function evaluation 实现，并记录仓库、commit、
license 和复用边界。
