# Dense

用明文权重计算全部 `Enc(W_i x)`，再用加密路由选择输出。所有 expert 共用输入
rotation，避免重复旋转造成不公平比较。

该方案保护路由，但执行全部 expert。统计包含线性变换和秘密输出选择。
实现位于 `dense.cpp`。
