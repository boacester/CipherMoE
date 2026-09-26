# MoE checkpoints

本目录保存阶段 A 使用的公开 MoE checkpoint。模型权重和 Hugging Face 下载缓存体积较大，
已由仓库根目录的 `.gitignore` 排除；本说明文件保留在 Git 中。

计划使用与 CryptoMoE 实验相对应的三个模型：

| 本地目录 | Hugging Face repository | 固定 revision | BF16 权重大小 |
| --- | --- | --- | ---: |
| `OLMoE-1B-7B-0924/` | `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | 13.84 GB |
| `deepseek-moe-16b-base/` | `deepseek-ai/deepseek-moe-16b-base` | `521d2bc4fb69a3f3ae565310fcc3b65f97af2580` | 32.75 GB |
| `Qwen1.5-MoE-A2.7B/` | `Qwen/Qwen1.5-MoE-A2.7B` | `1a758c50ecb6350748b9ce0a99d2352fd9fc11c9` | 28.63 GB |

可复现下载命令：

```bash
.venv/bin/python scripts/download_models.py
```

revision 的唯一来源是 `experiments/stage_a_models.json`。模型许可证分别以各模型仓库中的
license/model card 为准。
