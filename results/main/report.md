# CipherMoE 实验报告

生成时间：2026-09-26 15:35:31 +0800  
CPU：INTEL(R) XEON(R) PLATINUM 8592V  
OpenMP 线程：1；每项在线重复：3  
CKKS 安全级别：HEStd_128_classic；FHEW 参数：STD128。

## CKKS 矩阵实验

| case | method | median ms | rel. output error | ct-pt | ct-ct | rotations | selected-weight peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structured_n8_d16_r2 | public | 339.420 | 0.000e+00 | 16 | 0 | 15 | 0.00 |
| structured_n8_d16_r2 | dense | 1036.367 | 0.000e+00 | 128 | 8 | 15 | 0.00 |
| structured_n8_d16_r2 | select_weight_full | 1244.777 | 0.000e+00 | 128 | 16 | 15 | 16.00 |
| structured_n8_d16_r2 | select_weight_stream | 1158.889 | 0.000e+00 | 128 | 16 | 15 | 1.00 |
| structured_n8_d16_r2 | basis | 523.649 | 0.000e+00 | 48 | 2 | 15 | 0.00 |
| random_n8_d16_r2 | public | 331.525 | 0.000e+00 | 16 | 0 | 15 | 0.00 |
| random_n8_d16_r2 | dense | 1029.854 | 0.000e+00 | 128 | 8 | 15 | 0.00 |
| random_n8_d16_r2 | select_weight_full | 1244.776 | 0.000e+00 | 128 | 16 | 15 | 16.00 |
| random_n8_d16_r2 | select_weight_stream | 1159.424 | 0.000e+00 | 128 | 16 | 15 | 1.00 |
| random_n8_d16_r2 | basis | 524.845 | 8.673e-01 | 48 | 2 | 15 | 0.00 |
| random_n8_d16_r4 | public | 332.183 | 0.000e+00 | 16 | 0 | 15 | 0.00 |
| random_n8_d16_r4 | dense | 1027.184 | 0.000e+00 | 128 | 8 | 15 | 0.00 |
| random_n8_d16_r4 | select_weight_full | 1240.532 | 0.000e+00 | 128 | 16 | 15 | 16.00 |
| random_n8_d16_r4 | select_weight_stream | 1157.161 | 0.000e+00 | 128 | 16 | 15 | 1.00 |
| random_n8_d16_r4 | basis | 794.201 | 7.774e-01 | 96 | 4 | 15 | 0.00 |
| random_n8_d16_r8 | public | 330.409 | 0.000e+00 | 16 | 0 | 15 | 0.00 |
| random_n8_d16_r8 | dense | 1031.318 | 0.000e+00 | 128 | 8 | 15 | 0.00 |
| random_n8_d16_r8 | select_weight_full | 1309.204 | 0.000e+00 | 128 | 16 | 15 | 16.00 |
| random_n8_d16_r8 | select_weight_stream | 1166.859 | 0.000e+00 | 128 | 16 | 15 | 1.00 |
| random_n8_d16_r8 | basis | 1336.972 | 1.977e-15 | 192 | 8 | 15 | 0.00 |

## FHEW 标量联合查表

该微基准从分别加密的 1-bit expert ID 与 1-bit activation 开始；线性联合编码计入在线时间。

| variant | median ms | EvalFunc | inferred internal PBS |
| --- | ---: | ---: | ---: |
| fused_arbitrary | 811.755 | 1 | 2 |
| select_then_apply_arbitrary | 1622.922 | 2 | 4 |
| fused_negacyclic | 406.422 | 1 | 1 |
| select_then_apply_negacyclic | 808.570 | 2 | 2 |

## 主要结论

- 对精确 rank-2 共享结构，basis 相对 Dense 加速 1.98×，相对流式 Select-Weight 加速 2.21×，且不物化选中权重。
- 对随机权重，rank-2/rank-4 虽更快，但相对输出误差仍为 0.867/0.777；rank-8 达到近似精确时反而比 Dense 慢 1.30×。
- Select-Weight 的 block=1 将显式选中权重峰值从 16 MiB 降到 1 MiB，但密码学操作数不变，因此只解决存储而非计算。
- FHEW 半域负循环融合相对两阶段查表加速 1.99×；它证明标量融合有效，但尚未解决高维打包与 CKKS/FHEW 转换成本。

## 解释边界

- `relative_output_error` 对 basis 是低秩近似相对输出误差；其他方法应为 0（仅另报 CKKS 数值误差）。
- `selected-weight peak` 只统计显式选中权重密文载荷，不等同于进程 RSS。
- CKKS 从加密 one-hot 开始；FHEW 从加密 ID 开始，二者不是端到端同口径结果。
- 当前矩阵规模用于验证 primitive 和趋势，不代表完整 MoE 层或真实模型权重。
