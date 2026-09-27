# Stage B: multi-bit exact scalar FHEW（有限资源扫描）

## 口径

OpenFHE v1.5.1, STD128/GINX；单线程；key 和 model 是服务器公开参数，
只有 ID、activation 和输出是密文。ID 采用 ceil(log2 N) bits，
N 均为 2 的幂，因此该位宽没有非法码点；外部非法 ID 应在客户端加密前拒绝，
服务器不能对加密 ID 做明文合法性分支。

x 的 2-bit 定点编码为 code=0..3，对应 x=(code-2)/2；
w[e] 是 [-2,2] 中由公开公式 ((7e+3)%5)-2 得到的整数，
即结果的实数值为 (w[e]*(code-2))/2。为避免混淆，原始整数乘积
保存在 Z_16 中并以 [-8,7] 补码解码；所测输入下范围 [-4,4]，
没有溢出。Fused 直接在 Z_p 做多位联合编码 (4e+code)；
negacyclic 在 LUT 内使用 +5 offset 并在解密后去除；CMUX 输出 4 个 bit，
按 Z_16 解码。输入加密与解密不计入 online_ms；PBS 数据 OpenFHE 源码统计。

## 实测

| N | 路径 | p / ring | 检查/错误 | online median (ms) | PBS/CMUX | key (GiB) | LUT (KiB) | setup (s) | peak RSS (GiB) |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2 | fused | 16/4096 | 10/0 | 1680.6 | 2/0 | 9.26 | 32.0 | 101.0 | 9.55 |
| 2 | negacyclic | 16/4096 | 10/0 | 840.7 | 1/0 | 9.26 | 32.0 | 100.9 | 9.55 |
| 2 | cmux | 4/2048 | 10/0 | 7276.7 | 18/6 | 4.63 | 0.0 | 50.7 | 4.72 |
| 4 | fused | 16/4096 | 18/0 | 1683.7 | 2/0 | 9.26 | 32.0 | 101.5 | 9.55 |
| 4 | negacyclic | 32/8192 | 18/0 | 1764.0 | 1/0 | 18.52 | 64.0 | 202.2 | 19.60 |
| 4 | cmux | 4/2048 | 6/0 | 14518.5 | 36/12 | 4.63 | 0.0 | 50.6 | 4.72 |
| 8 | fused | 32/8192 | 10/0 | 3516.9 | 2/0 | 18.52 | 64.0 | 202.7 | 19.60 |
| 8 | cmux | 4/2048 | 5/0 | 29050.5 | 72/24 | 4.63 | 0.0 | 50.4 | 4.72 |
| 16 | cmux | 4/2048 | 5/0 | 47206.8 | 117/39 | 4.63 | 0.0 | 50.7 | 4.72 |

## 未完成 / 失败项

- `32:cmux:4:1`: Command '['/home/liupeixin/workspace2026/CipherMoE/build/stage_b_scalar', '32', 'cmux', '4', '1']' timed out after 360 seconds
- `64:cmux:4:1`: Command '['/home/liupeixin/workspace2026/CipherMoE/build/stage_b_scalar', '64', 'cmux', '4', '1']' timed out after 360 seconds
- 以上只有 N=2 所有路径以及 N=4 fused/negacyclic（若列出）做了完整枚举；
  其他为抽样，无穷举正确性结论。
- TFHE-rs WOPBS-only 的 legacy 参数已独立实测，但未证明与 OpenFHE STD128 安全等价；多输出 vertical packing 仍未实测。
- Fused 和 CMUX 的 ring dimension 不同；本表可比安全级别与输入输出语义，
  但延迟不能被解释成同一参数下的纯算法加速比。
- CMUX 当前是输出 bit 的 Boolean truth-table/CMUX tree；它使用 N 个公开
  leaf functions 并对相同子表达式做缓存，不是可扩展的矩阵方案。
- CMUX 输入是客户端分别加密的 ID bits 和 x bits，没有把单个 `Enc(e)` 同态分解成 bits 的成本；输出 4 个 bit 密文，而 fused LUT 输出 1 个标量密文。
- `evaluation_key_bytes` 为 refresh + switching keys 的 portable binary 序列化大小；
  LUT 是 `q * sizeof(NativeInteger)`，不是 evaluation key；peak RSS 是进程高水位。

## 阶段决策

当前仅证明部分 exact 标量原语可计算；TFHE-rs WOPBS-only 已单列探索，未完成同口径安全参数对照与所有 N 的穷举。
直接联合 LUT 的明文空间至少为 N*4，要求 ring dimension >= 256*N*4；
依赖半域 negacyclic 时还要翻倍。Key storage 与 setup 已成为矩阵扩展的关键障碍，
不能将本结果宣称为完整 SelectApply 或 Stage C 的 go。先评估任意权重矩阵
所需 public data / LUT / evaluation-key 信息下界，再验证复用选择的接口。

## 复现

```bash
cmake --build build --target stage_b_scalar -j8
.venv/bin/python experiments/run_stage_b.py
```

原始配置见 `metadata.json`，每项原始结果见 `raw/`，失败记录见 `failures.json`。

## TFHE-rs WOPBS-only（探索性参数）

固定 Zama tfhe-rs `079f46d08ffb8763de54b7794e850ddbd98fe63b` (BSD-3-Clause-Clear)，
`shortint` + `experimental` 的 `WopbsKey::new_wopbs_key_only_for_wopbs`、
`generate_lut_without_padding` / `programmable_bootstrapping_without_padding`。
选择 ID 与 x 分开加密，在线先线性合成为 `4e+code`，再做 WOPBS。
这是 single-output WOPBS，并非多输出 vertical packing 对照。

| N | p | 检查/错误 | online median (ms) | key (GiB) | LUT (KiB) | setup (s) | peak RSS (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 10/0 | 294.6 | 0.44 | 8.0 | 6.3 | 0.57 |
| 4 | 16 | 18/0 | 290.7 | 0.44 | 8.0 | 6.3 | 0.57 |
| 8 | 32 | 10/0 | 691.8 | 0.72 | 8.0 | 11.9 | 0.93 |
| 16 | 64 | 10/0 | 843.6 | 0.72 | 8.0 | 12.0 | 0.93 |
| 32 | 128 | 10/0 | 972.7 | 0.72 | 8.0 | 11.9 | 0.93 |
| 64 | 256 | 10/0 | 1141.5 | 0.72 | 8.0 | 12.0 | 0.93 |

TFHE-rs 的 N=2/4 各穷举了 8/16 个输入组合；其余规模为抽样。
TFHE-rs 表采用 *legacy WOPBS-only* 参数；其安全性未与 128-bit 主表
同口径认证，且多输出 vertical packing 未测试，因此不能从两个表推导跨库加速倍数。
key_bytes 是 WopbsKey 的 bincode 序列化大小（含内部 server keys），
LUT 单列；固定权重、定点域与 OpenFHE 实验一致。

## 复用原语与边界

| 源码 | 固定版本/许可 | 复用 API | 限制 |
| --- | --- | --- | --- |
| OpenFHE `thirdparty/openfhe/src` | v1.5.1 / `1306d14`, BSD-2-Clause | `EvalFunc`、`EvalBinGate(CMUX)`、`GetMaxPlaintextSpace` | 一般 LUT 2 次 PBS；CMUX 内部 3 次 NAND/PBS；p <= q/256 |
| Zama tfhe-rs（Cargo.lock） | `079f46d`, BSD-3-Clause-Clear | `WopbsKey::new_wopbs_key_only_for_wopbs`、`programmable_bootstrapping_without_padding` | `experimental` feature；legacy 参数安全级别待核验；vertical packing 未实测 |

上述均调用官方提供的原语，未自行实现 PBS 内核。
默认权重表仅有五种数值；对于重复权重，CMUX 缓存会复用布尔子表达式，
因此成本不能外推到高精度真实模型权重。

## 资源下界与失败边界

任意 public scalar table 至少包含 N 个独立权重；本实验的 fused LUT
需要容纳 4N 个 `(e,x)` 输入。OpenFHE v1.5.1 的 `GetMaxPlaintextSpace=q/256`，
因此该直接编码至少需要 `p>=4N, q>=1024N`；若用负循环半域扩展，
需要 `p>=8N, q>=2048N`。由 q=4096/8192 两点实测 key 大小几乎翻倍，
按 ring dimension 线性外推（**非实测**）：

| N | fused q 下限 | 估计 OpenFHE key (GiB) | 32 GiB 进程预算 |
| ---: | ---: | ---: | --- |
| 16 | 16384 | 37.0 | 超出 |
| 32 | 32768 | 74.1 | 超出 |
| 64 | 65536 | 148.2 | 超出 |

这些估计仅适用于这里的 OpenFHE 直接联合编码，不能用于宣称 TFHE-rs
也具有相同 key 增长。对任意权重矩阵，公开模型至少需存放
`N*d_in*d_out*weight_bits/8` 字节；若逐元素/输出独立执行本标量原语，
LUT/PBS 成本还会乘上元素/输出数。必须先实测一次选择的多输出复用，
才有进入矩阵 Stage D 的依据。

## 自定义公开权重表

- `tfhe_wopbs_legacy` `[2,-1,-2,0]`: 检查 17、错误 0。
- `openfhe_fused` `[2,-1]`: 检查 9、错误 0。
- `openfhe_cmux` `[2,-1]`: 检查 9、错误 0。

复现：`.venv/bin/python experiments/test_stage_b_tables.py`。

## 大 N 的低样本 CMUX 探针

主扫描的 4 次检查 + 1 次计时设为 360 秒/项；下表是独立运行的
1 次检查 + 1 次计时（500 秒/项），不能与多次重复的 median 同等解释。

| N | 验证/错误 | 单次在线 (ms) | PBS |
| ---: | ---: | ---: | ---: |
| 32 | 2/0 | 71423.5 | 177 |
| 64 | 2/0 | 95747.3 | 237 |

复现：`.venv/bin/python experiments/probe_stage_b_cmux.py`。

复现：`RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR=$PWD/build/tfhe-target cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml`，
然后运行 `.venv/bin/python experiments/run_stage_b_tfhe.py`。
