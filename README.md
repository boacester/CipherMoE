# CipherMoE

探索非交互式 FHE 下的私有 MoE 路由和 Select-and-Apply。核心任务是用加密的
路由、加密的 activation 和服务器的明文权重，计算 `Enc(W_e x)`。

## 目录

```text
baseline/
  Dense/              计算全部 expert，再秘密选择输出
  Select_Weight/      秘密选择加密权重再计算；包含分块消融
  Routing_Publicity/  路由公开时的性能参照
proposal/             exact FHEW 主方向与 structured low-rank 候选
models/               真实 MoE checkpoint（权重不纳入 Git）
experiments/          共用实验代码和批量运行入口
documents/            初始想法与实验报告
results/              配置、原始指标和图表
scripts/              依赖安装和安装检查
thirdparty/           外部依赖的本地源码和安装目录
```

初始方案见 [documents/idea.md](documents/idea.md)，对照约定见
[baseline/README.md](baseline/README.md)。实验从已提供的加密路由开始，
尚不代表完整 router 或 MoE 的端到端性能。

## 构建和运行

```bash
bash scripts/install_openfhe.sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-experiments.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/thirdparty/openfhe/install"
cmake --build build --parallel 8
.venv/bin/python experiments/run.py
```

OpenFHE 固定为 `v1.5.1`，本地安装目录为 `thirdparty/openfhe/install`。
详细说明见 [thirdparty/README.md](thirdparty/README.md)。

## 分阶段实验

阶段 A 使用固定 revision 的真实 MoE checkpoint，先做明文权重与 calibration activation
结构分析，不运行 FHE：

```bash
.venv/bin/python scripts/download_models.py
.venv/bin/python experiments/capture_moe_activations.py
OPENBLAS_NUM_THREADS=64 .venv/bin/python experiments/analyze_moe_weights.py \
  --calibration-dir results/stage_a/calibration
.venv/bin/python experiments/report_stage_a.py
```

模型 revision 记录在 `experiments/stage_a_models.json`，模型权重保存在 `models/` 且不纳入
Git。相关工作总结见 [documents/related_work.md](documents/related_work.md)，实现与验证计划见
[documents/implement.md](documents/implement.md)。

阶段 B 单独运行 multi-bit exact scalar 实验；OpenFHE 主表采用 STD128，TFHE-rs 的
WOPBS-only 参数仍属实验性且尚未做同口径安全级别核验：

```bash
cmake --build build --target stage_b_scalar -j8
.venv/bin/python experiments/run_stage_b.py
RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR="$PWD/build/tfhe-target" \
  cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml
.venv/bin/python experiments/run_stage_b_tfhe.py
.venv/bin/python experiments/test_stage_b_tables.py
.venv/bin/python experiments/probe_stage_b_cmux.py
.venv/bin/python experiments/run_stage_b_tfhe.py --report-only
```

逐项原始数据和资源限制见 [results/stage_b/report.md](results/stage_b/report.md)。
这些是秘密 ID 与秘密标量输入的实验，不代表 packed matrix 或完整 MoE 推理。

阶段 C 验证同一秘密输入上的多输出 LUT 复用；阶段 D 对 `N=2` 的矩形矩阵
做 tiled 有限域原型。E/F 因模型尺度路径与混合接口尚未成立而未运行：

```bash
RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR="$PWD/build/tfhe-target" \
  cargo build --release --manifest-path experiments/tfhe_stage_b/Cargo.toml \
  --bin stage_c --bin stage_d
.venv/bin/python experiments/run_stage_c.py
.venv/bin/python experiments/run_stage_d.py
```

实测结果、资源门槛和未执行原因见 [Stage C](results/stage_c/report.md)、
[Stage D](results/stage_d/report.md)、[Stage E](results/stage_e/report.md) 与
[Stage F](results/stage_f/report.md)。其中 Stage D 仅为小尺寸原型，
不代表通过模型尺度 SelectApply 验收。

## 当前实验结果

已完成的单线程、三次重复主实验见 [results/main/report.md](results/main/report.md)，
原始 JSON、汇总 CSV、模型输入与延迟图位于同一目录。`experiments/run.py` 默认
重跑主实验；快速检查可使用 `--profile smoke --repeats 1 --skip-lut`。历史报告中的
`basis` 标签对应现已改名的 `LowRank_Structured`，不代表 exact 主方案。
