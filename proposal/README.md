# Select-and-Apply proposals

本目录把两个性质不同的候选严格分开。项目最终目标是对任意 pretrained MoE 原始权重实现：

```text
SelectApply(Enc(e), Enc(x), {W_i}) -> Enc(W_e x)
```

要求路由和 activation 对服务器隐藏，不重新训练或修改模型，不依赖 experts 的特殊共享
结构，不物化完整 `Enc(W_e)`，并避免计算全部 experts。

这里的 exact 指精确选择并应用原始 `W_e`，没有跨-expert 结构近似；CKKS/定点编码带来的
数值误差仍需单独测量，不属于模型结构误差。

| 目录 | 定位 | exact / arbitrary pretrained MoE |
| --- | --- | --- |
| `FHEW_SelectApply/` | 以 FHEW/TFHE programmable bootstrapping 实现 encrypted control；当前只有标量联合 LUT 原型 | **主研究方向**；目标是 exact，但当前原型尚未扩展到向量/矩阵 |
| `LowRank_Structured/` | `W_e ≈ Σ_j c[e,j] B_j` 的共享低秩基 | 仅 structured / approximate candidate 和 baseline；`r<N` 一般不 exact |

`LowRank_Structured` 的实验不能被用来证明 exact Select-and-Apply 已实现。若真实 checkpoint
在低误差下普遍要求 `r≈N`，应停止把它作为性能主线，只保留为可选结构优化。
