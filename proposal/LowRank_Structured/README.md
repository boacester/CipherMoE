# LowRank Structured candidate

该候选假设不同 experts 的权重可由少量共享基近似：

```text
W_e ≈ Σ_{j=1}^r c[e,j] B_j
```

它秘密选择系数，应用 `r` 个明文基矩阵，再对基输出做密文门控。只有当 `r << N` 时才
可能有明显收益；对任意 pretrained weights 没有这种保证。`r<N` 时通常是近似方法，
只有 weights 恰好位于该子空间时才 exact。

因此本目录是 structured-model baseline / optional optimization，不是 exact Select-and-Apply
的定义或主实现。`low_rank_basis.cpp` 是当前 CKKS 小规模 primitive。
