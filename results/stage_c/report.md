# Stage C：同一秘密输入的多输出复用（有限域实测）

## 参数与正确性

固定 Zama TFHE-rs `079f46d08ffb8763de54b7794e850ddbd98fe63b`
（BSD-3-Clause-Clear），CPU 单线程，参数
`V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128`：message=8、
carry=8，accumulator 可用明文区间 64，参数名的 `2M128` 指单次失败概率
约 2^-128，**不代表**本阶段已证明与 Stage B OpenFHE STD128
有可比的安全级别。统一以 `Enc(e)`、`Enc(code)` 形成 `4e+code`，
`e in [0,N)`, `code in [0,4)`, `x=code-2`；公开权重属于 [-2,2]，
每个输出函数的权重行不同（程序运行时断言），加密输出为
`w_k[e]*x mod 16`，范围 [-4,4] 没有 signed Z_16 溢出。
服务端只使用输入合法性所给的公开 degree 上界，不因秘密 ID 修改电路轨迹。

比较：相同联合输入上 `m` 次 `generate_lookup_table/apply_lookup_table`
与一次 `generate_many_lookup_table/apply_many_lookup_table`。每种路径对
全部 `4N` 个 `(e,code)` 组合、全部 `m` 个输出解密对照明文结果；
另有 3 次计时重复，轮换求值先后顺序。输出为 `m` 个独立 LWE 密文，
不是 CKKS slots 或一个 packed RLWE 密文。下列在线中位数仅包括 LUT 求值，
不含加密、联合编码、解密；`encryption_ms`、`decryption_ms` 为累计值，
联合编码未单独计时，因此不能把这些中位数称为端到端请求延迟。

## 实测（2026-09-27）

| N | 输出 m | 穷举输入 / 错误输出 | 独立中位 (ms) | many-LUT 中位 (ms) | 独立 PBS / many blind rotation | 独立 / many LUT (KiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 8 / 0 | 200 | 200 | 1 / 1 | 128 / 128 |
| 2 | 2 | 8 / 0 | 455 | 199 | 2 / 1 | 256 / 128 |
| 2 | 4 | 8 / 0 | 835 | 195 | 4 / 1 | 512 / 128 |
| 2 | 8 | 8 / 0 | 1653 | 199 | 8 / 1 | 1024 / 128 |
| 4 | 1 | 16 / 0 | 204 | 196 | 1 / 1 | 128 / 128 |
| 4 | 2 | 16 / 0 | 399 | 234 | 2 / 1 | 256 / 128 |
| 4 | 4 | 16 / 0 | 844 | 196 | 4 / 1 | 512 / 128 |
| 8 | 1 | 32 / 0 | 199 | 198 | 1 / 1 | 128 / 128 |
| 8 | 2 | 32 / 0 | 396 | 197 | 2 / 1 | 256 / 128 |

9 项 `failures.json` 均无异常；每项单独生成 key，序列化的 server
key 为 989461560 bytes（0.92 GiB）、keygen 约 29--30 秒，
进程峰值约 0.99 GiB。两种路径的 LUT 预处理合计约 0.12--0.51 ms；
未拆成各方法独立预处理，也不能从这些数字推出生产部署的 setup 分摊。
LUT 体积是公开 accumulator 内 `u64` 字节数，不含对象元数据。
原始每次延迟、预处理、内存、密钥和验证数量均在 `raw/`。

## 复用机制、门槛与未测项

该版本 many-LUT 的 `fill_many_lut_accumulator` 把一个多项式按输出
函数数目切段，输入最大 degree 为 `64/m-1`；`standard.rs` 实现先进行
一次 keyswitch 与 blind rotation，再对每个输出做 sample extraction。
因此同一参数下复用一次 encrypted selection 的**边际输出**不需要
新 blind rotation，实测 `m=1 -> 8` 的 LUT 求值耗时基本不随 m 增长。
但必要条件 `4*N*m <= 64` 立即限制规模：`N=8` 只能 `m<=2`，
`N=64` 连一个输出都无法表示。这是以输入域容量换共享计算量，
不能将 8 个小 N 输出的延迟外推到 64 experts 或真实输出维度。

- OpenFHE STD128 的独立 `EvalFunc` 路径在 Stage B 测过单输出；
  **未**在同一加密输入上做多输出重复的等口径扫描。
- OpenFHE 实验性 `EvalMVBPrecompute/EvalMVB` 的输入为系数编码
  RLWE/CKKS，预计算共享到复指数幂后，仍需每个函数单独调用 `EvalMVB`；
  本轮只核查固定版本的官方源码/说明，**未实测**预计算和边际输出耗时，
  不能与 TFHE-rs shortint 的数字直接比较。
- TFHE-rs WOPBS `circuit_bootstrapping_vertical_packing` 的
  bit-extraction/输出布局链路未接通，**未实测**；Stage B legacy WOPBS
  单输出参数未获同安全级别证明，不能用其单输出速度填充本表。
- 没有 CKKS 输出 slots、模型权重高精度编码、向量输入或模型尺度测试。

**阶段决策：小 N 上边际 PBS 数达到预期；Stage C 全口径 go 暂缓。**
缺少 OpenFHE MVB 与 TFHE-rs WOPBS vertical packing 对照，而且固定输入
域容量在大 N 下失败。Stage D 的结果是并行开展的**探索性小矩阵原型**，
不是通过 Stage C 门槛后的可扩展矩阵验收。

## 复现

```bash
RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR=$PWD/build/tfhe-target \
  cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml --bin stage_c
.venv/bin/python experiments/run_stage_c.py --repeats 3 --timeout 600
```

实验配置及约束见 `metadata.json`，各配置原始结果见 `raw/`，汇总见
`summary.json`，失败记录见 `failures.json`。修正输出函数互异性后仅需
重跑受影响的项时使用 `--only 2,8`。
