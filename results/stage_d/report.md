# Stage D：小尺寸矩形 tiled SelectApply（探索性原型）

## 口径

TFHE-rs 固定 commit `079f46d08ffb8763de54b7794e850ddbd98fe63b`
（BSD-3-Clause-Clear），`V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128`
（message/carry 各 8，参数名中的 `2M128` 是失败概率口径，**不是**与
OpenFHE STD128 的安全性等价证明），单线程、32 GiB 进程限额。
expert ID `e in {0,1}` 和每个向量分量 `x_j=code_j-2`（code=0..3）分别加密。
公开权重 `W[e,row,col] in {-1,0,1}`；生成式与显式数组都经过验证。
服务端构造 `4e+code_j`，为每个输入分量和输出 tile 用一个
`generate_many_lookup_table` / `apply_many_lookup_table` 对多个输出取值；
再对不同输入分量的输出密文求和。返回 `d_out` 个独立 LWE 密文，
**不是** CKKS slots。明文参考检查 Z_16 结果，所有测试的真实整数结果位于
[-6,6]，不存在 Z_16 signed 解码的溢出。

两条路径均使用相同密文输入、相同公开矩阵；baseline 为每个 `(col,row)`
单独执行一次 `apply_lookup_table`。所列在线中位数仅覆盖 LUT 求值核函数，
**不包含**加密、联合编码、密文求和或解密；这些步骤在 raw JSON 中分别
记录了累计 `encrypt_ms`（加密及联合编码）、`combine_ms` 和 `decrypt_ms`，
不能把核函数计时称作完整请求延迟。每项均另行生成 key（约 29.7 s），
预处理开销见 raw JSON。输出位置不依赖秘密 ID，没有构造完整 `Enc(W_e)`，
也没有计算 N 个完整 MatMul。

## 实测（2026-09-27）

| d_in | d_out | tile | 检查/错误 | 独立 / tiled 核耗时中位 (ms) | 独立 PBS / tiled blind rotation | 独立 / tiled LUT (KiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 2 | 32 / 0，完整枚举 | 1222 / 812 | 6 / 4 | 768 / 512 |
| 3 | 4 | 2 | 16 / 0，抽样 | 2439 / 1197 | 12 / 6 | 1536 / 768 |
| 3 | 4 | 4 | 16 / 0，抽样 | 2431 / 602 | 12 / 3 | 1536 / 384 |
| 2 | 3 | 2，显式权重 | 16 / 0，抽样 | 1241 / 835 | 6 / 4 | 768 / 512 |

显式权重按 expert、row、col 顺序为
`[1,0,-1,1,0,1,0,-1,1,-1,1,0]`。两个 `3 -> 4` 测试
覆盖不同 activation 码点，但不是 `2*4^3=128` 个输入的穷举。
key 序列化大小每项 989461560 bytes（0.92 GiB），进程峰值约 0.99 GiB；
每个明文 LUT accumulator 为 128 KiB，所列 LUT 是 accumulator 的总字节数，
不包含对象元数据。`failures.json` 为空。

## 算法及扩展边界

所测 tiled 路径对每个输入分量和每个输出 tile 执行一次 blind rotation，
总数 `d_in * ceil(d_out/tile)`；之后另需 `d_out*(d_in-1)` 次 LWE 加法。
独立 baseline 需要 `d_in*d_out` 次 PBS。公开模型自身至少有
`N*d_in*d_out` 个权重，而这种实现还需要
`d_in*ceil(d_out/tile)` 个完整 LUT accumulator。例如按**当前固定参数和
128 KiB accumulator**、`d_in=2048,d_out=1024,tile=4` 仅计算个数就得到
524288 次 blind rotation 与 64 GiB LUT 存储。这是算术外推，**不是**
该真实维度或 `N=64` 的运行测试；在当前参数下 `N=64` 连一个联合
`(e,x)` 域都无法编码（需要 256 个输入码点，而容量仅 64）。

这种方法仍为每个 `x_j` 再做每个 tile 的 LUT 求值；输入域容量条件为
`4*N*tile <= message_modulus*carry_modulus`。现有 V1_8 公开固定参数
最大测试到 4+4 bits（容量 256），在 `N=64` 时甚至 `tile=2` 也需要
512 个码点。不能把小 `N` 的提速直接迁移到 Stage A 的真实模型。
尚未测试 input-stationary、diagonal/BSGS 或 packed RLWE 布局，也未把
CKKS↔FHEW 转换计入；仅支持权重集合 {-1,0,1} 和 `N=2` 的 ID。

**阶段决策：小矩阵功能验证通过，模型尺度 no-go / Stage E 暂缓。**
该结果不是完整 SelectApply 的可扩展性证明。须先找到能在大 N 下兼顾
多输出复用、公开参数存储和 packed 输出的原语，再运行混合路径。

## 复现

```bash
RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR=$PWD/build/tfhe-target \
  cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml --bin stage_d
.venv/bin/python experiments/run_stage_d.py
```

原始配置见 `metadata.json`，每项原始计时和计数见 `raw/`，汇总见
`summary.json`；脚本会保存失败记录到 `failures.json`。
