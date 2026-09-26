# CipherMoE 实验报告

生成时间：2026-09-26 15:22:06 +0800  
CPU：x86_64  
OpenMP 线程：1；每项在线重复：1  
CKKS 安全级别：HEStd_128_classic；FHEW 参数：STD128。

## CKKS 矩阵实验

| case | method | median ms | rel. output error | ct-pt | ct-ct | rotations | selected-weight peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structured_n4_d8_r2 | public | 158.017 | 0.000e+00 | 8 | 0 | 7 | 0.00 |
| structured_n4_d8_r2 | dense | 344.065 | 0.000e+00 | 32 | 4 | 7 | 0.00 |
| structured_n4_d8_r2 | select_weight_full | 437.153 | 0.000e+00 | 32 | 8 | 7 | 8.00 |
| structured_n4_d8_r2 | select_weight_stream | 411.114 | 0.000e+00 | 32 | 8 | 7 | 1.00 |
| structured_n4_d8_r2 | basis | 269.372 | 0.000e+00 | 24 | 2 | 7 | 0.00 |
| random_n4_d8_r2 | public | 157.364 | 0.000e+00 | 8 | 0 | 7 | 0.00 |
| random_n4_d8_r2 | dense | 344.953 | 0.000e+00 | 32 | 4 | 7 | 0.00 |
| random_n4_d8_r2 | select_weight_full | 426.908 | 0.000e+00 | 32 | 8 | 7 | 8.00 |
| random_n4_d8_r2 | select_weight_stream | 408.572 | 0.000e+00 | 32 | 8 | 7 | 1.00 |
| random_n4_d8_r2 | basis | 267.047 | 8.588e-01 | 24 | 2 | 7 | 0.00 |

## 解释边界

- `relative_output_error` 对 basis 是低秩近似相对输出误差；其他方法应为 0（仅另报 CKKS 数值误差）。
- `selected-weight peak` 只统计显式选中权重密文载荷，不等同于进程 RSS。
- CKKS 从加密 one-hot 开始；FHEW 从加密 ID 开始，二者不是端到端同口径结果。
- 当前矩阵规模用于验证 primitive 和趋势，不代表完整 MoE 层或真实模型权重。
