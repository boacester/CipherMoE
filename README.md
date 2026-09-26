# CipherMoE

探索非交互式 FHE 下的私有 MoE 路由和 Select-and-Apply。核心任务是用加密的
路由、加密的 activation 和服务器的明文权重，计算 `Enc(W_e x)`。

## 目录

```text
baseline/
  Dense/              计算全部 expert，再秘密选择输出
  Select_Weight/      秘密选择加密权重再计算；包含分块消融
  Routing_Publicity/  路由公开时的性能参照
src/                  Select-and-Apply 候选及优化
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

## 当前实验结果

已完成的单线程、三次重复主实验见 [results/main/report.md](results/main/report.md)，
原始 JSON、汇总 CSV、模型输入与延迟图位于同一目录。`experiments/run.py` 默认
重跑主实验；快速检查可使用 `--profile smoke --repeats 1 --skip-lut`。
