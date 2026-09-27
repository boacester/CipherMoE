# P0：只由加密 expert ID 驱动的多输出选择

## 问题与口径

本轮独立测试 `MultiOutputSelect(Enc(e), public {V_i}) -> Enc(V_e)`，
**不输入 activation**，不扩展 `(e,x)` 联合 LUT。公开 `V_i` 是可逐项变化的
4-bit 无符号向量（值 0..15，用 `u8` 存放），默认由固定哈希生成；
额外对显式 JSON 公开向量表做正确性检查。对这个有限域的 selection
语义没有跨 expert 低秩或稀疏假设。P0 诊断中输出完整 `Enc(V_e)` 是允许的，
**不能**将它当作最终禁止物化完整 `Enc(W_e)` 的 SelectApply 构造。

固定 TFHE-rs `079f46d08ffb8763de54b7794e850ddbd98fe63b`
（BSD-3-Clause-Clear），参数
`V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128`：
message=8、carry=8，总 LUT 输入容量 64，GLWE polynomial=8192，
LWE dimension=1006，PBS decomposition level=2，源码参数给出
`log2_p_fail=-128.115`。这是失败概率，**不是**与 OpenFHE STD128
独立验证的密码学安全等价。客户端以公开上界 `N` 和 `64/N` 配置
`MessageModulus/CarryModulus`，将 ID 加密成**一个** shortint ciphertext；
服务器对固定公开向量预计算 LUT，同一密文上做选择。

CPU：Intel Xeon Platinum 8592V；`RAYON_NUM_THREADS=1`，
32 GiB 进程地址空间限制。每个 `N` 只生成**一套**密钥，跨全部 `m`
复用；密钥生成、客户端加密、模型生成、LUT 预计算、求值与解密单列。
`T(m)` 是产生 `m` 个输出 LWE 的服务器 LUT 求值时间中位数，
不含其他阶段。每项重复计时两次；`N=2/8` 全部 expert ID 穷举，
`N=64` 检查 0、32、63 三个 ID（**不是**完整 64 个 ID 穷举），
每个受检 ID 的 `m` 个输出均与公开参考值逐项比较，计时重复亦解密检查。

## 实测（2026-09-27）

| m | N=2: T(ms) / rotations | N=8: T(ms) / rotations | N=64: T(ms) / rotations |
| ---: | ---: | ---: | ---: |
| 1 | 241.5 / 1 | 219.3 / 1 | 226.2 / 1 |
| 4 | 213.2 / 1 | 197.2 / 1 | 852.0 / 4 |
| 8 | 200.7 / 1 | 201.7 / 1 | 1641.8 / 8 |
| 16 | 200.1 / 1 | 392.9 / 2 | 3172.7 / 16 |
| 32 | 197.1 / 1 | 788.5 / 4 | 6417.3 / 32 |
| 64 | 396.7 / 2 | 1571.6 / 8 | 12668.0 / 64 |
| 128 | 789.7 / 4 | 3177.1 / 16 | 25417.6 / 128 |
| 256 | 1632.0 / 8 | 6306.7 / 32 | 51091.6 / 256 |
| 512 | 3168.1 / 16 | 12650.6 / 64 | 101838.4 / 512 |

`m=512` 时 `T(m)/m` 依次为 **6.19 / 24.71 / 198.90 ms**；
相比各自 `T(1)` 的 **241.51 / 219.34 / 226.19 ms**，
只有小 `N` 出现持续摊销。对于 `N=64`，`m=4/8/16` 的独立
`apply_lookup_table` 基线另有**实测**中位耗时 836.6/1618.8/3183.7 ms，
对应共享路径 852.0/1641.8/3172.7 ms，未表现出实质收益。
`m>16` 的逐输出独立 baseline **没有实测**，不能把 `m*T(1)`
称作实测基线。全部 27 个默认配置零错误，`N=2,m=8` 的显式权重
JSON 用例也零错误（2 个 ID 的每个输出均检查）；`failures.json` 为空。

## 操作数与存储

固定版本 `shortint/engine/mod.rs::fill_many_lut_accumulator` 的输入
degree 上界为 `64/function_count - 1`。因 ID 范围为 `0..N-1`，
单 accumulator 最多容纳 `64/N` 个输出，需 `ceil(m/(64/N))`
个 accumulator/keyswitch/blind rotation；`standard.rs` 实现每个
accumulator 旋转一次，再为其中每个输出做一次 LWE sample extraction。
这解释了表中 `N=64` 从第一个输出起就是**每输出一次旋转**。
源代码给出的计算结构是每次旋转至多 1006 次基于 mask 的 external
product（只在模切换后的 mask 非零时执行，内部使用 2 级分解）；
`N=64,m=512` 的上界为 515072 次。这是**源码计数上界**，
不是已插桩得到的 exact external-product 实测。源码中的 mask 条件
取决于公开密文随机性；未进行机器级 constant-time 旁路审计。
P0 不进行 WOPBS、CKKS 乘法/rotation 或 scheme switching（次数为 0）。

| N, m=512 | 公开权重 (`u8`) | LUT accumulators | LUT 字节 | 请求密文 | 响应密文 | 评估密钥 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1024 B | 16 | 2 MiB | 65616 B | 33595392 B | 989461560 B |
| 8 | 4096 B | 64 | 8 MiB | 65616 B | 33595392 B | 989461560 B |
| 64 | 32768 B | 512 | 64 MiB | 65616 B | 33595392 B | 989461560 B |

每项 LUT accumulator 为 128 KiB（不含对象元数据）；客户端请求单个
ID 密文，响应为 `m` 个**独立** LWE 密文，并非紧凑 RLWE/CKKS slots。
密钥序列化体积约 0.92 GiB，keygen 每个 `N` 约 30 秒，
`N=64,m=512` LUT 预处理约 28 ms，进程峰值约 1.02 GiB。
评估密钥传输是额外 setup 成本，表中请求/响应字节不含它。
各输出密文长度由固定参数与维度决定；选择代码不依据秘密 ID
改变 LUT 数、PBS 数、输出数或公开内存布局。但未审计底层实现对
公开密文 mask 的条件分支/微架构侧信道，不能声称形式化恒时证明。

## 阶段结论

**P0 结论：此固定参数的 TFHE-rs shortint many-LUT 在 N=64 时 no-go。**
它确实能在 `N=2/8` 对多个输出共享一次 blind rotation，
但在 64 experts 下 `m=512` 实测 101.84 秒和 512 次旋转，
没有所需的 encrypted selector 摊销；不能继续把分块 scalar LUT
扩大成真实模型矩阵，亦不能因此宣称 P1/P2 已通过。
这只是**该原语与参数**的负面结果，不证明任何 FHE packed
selector 都不可行。下一步应先验证 packed RLWE/GLWE CMUX
或 OpenFHE 的 RLWE/CKKS `EvalMVB` 等不同表示，明确 bit extraction、
circuit bootstrap、key switch、CKKS 转换与 tile 布局的全部成本，
并禁止退化成 `Enc(W_e)` 物化或 N 个完整 expert MatMul。
TFHE-rs 的 `core_crypto` 有 GLWE CMUX 与 circuit-bootstrap/vertical-packing
入口，OpenFHE v1.5.1 `EvalMVBPrecompute/EvalMVB` 可共享 RLWE/CKKS
中间计算；两者均**未在本轮实现或测速**，不能推定有收益。

## 复现与原始记录

```bash
RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR=$PWD/build/tfhe-target \
  cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml --bin p0_selector
.venv/bin/python experiments/run_p0_selector.py --timeout 1500 --repeats 2
```

参数和线程见 `metadata.json`；全部 28 项原始时延、加密/解密开销、
响应大小及 key/LUT 字节数见 `raw/`、`summary.json`。
显式向量输入为 `experiments/p0_custom_vectors.json`。
