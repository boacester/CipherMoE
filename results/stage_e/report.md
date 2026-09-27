# Stage E：CKKS/FHEW 混合接口（未执行）

状态：**前置阶段未通过，未运行混合路径，也未生成性能或精度数据。**
这不是 Stage E 的完成报告或混合方案 no-go 证明。

## 源码核查

- OpenFHE v1.5.1 (`1306d14`) 的 `scheme-switching.cpp` 展示
  `EvalCKKStoFHEW` / `EvalFHEWtoCKKS`；其首个示例使用 `TOY` 和
  `HEStd_NotSet`，不能替代统一安全级别的测量。
- 同版本 `CKKS_FUNCTIONAL_BOOTSTRAPING.md` 的 `EvalMVBPrecompute/EvalMVB`
  使用系数编码 RLWE 输入和 CKKS 中间态；它不是 Stage C 使用的 TFHE-rs
  shortint 密文可直接传入的接口。
- Stage D 只在 `N=2` 的 2/3 维输入上验证了有限域矩阵内核；还没有
  大维度可用的 CKKS-compatible 控制对象或输出布局。

## 尚缺的实验

1. 在同一安全参数与明确定点范围内，测量 CKKS -> FHEW/RLWE -> CKKS 的
   双向转换和旋转、bootstrap 次数，以及各步骤数值误差。
2. 实现其中一种不会生成 one-hot 再回退 Dense CKKS 的混合算子；若选择
   TFHE-rs Stage D 内核，需要设计并验证与 OpenFHE CKKS 的密钥/密文互操作，
   不能按字节格式相似来假设可以直接交换。
3. 与相同参数的 Public Routing、Dense-FHE 和 Select-Weight 比较。

阶段门槛：**暂缓**。Stage C 尚未实测 OpenFHE MVB 与 TFHE-rs WOPBS
vertical packing；Stage D 没有证明模型尺度矩阵路径可用，因此不能声称
本阶段已运行或进入完整 MoE 集成。
