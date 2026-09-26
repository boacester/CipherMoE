# OpenFHE 本地安装

固定版本为 `v1.5.1`，提交为
`1306d14f8c26bb6150d3e6ad54f28dfe1007689e`，包含该版本的上游子模块。

```text
thirdparty/
  openfhe/
    src/          上游源码
    build/        编译后的库和部分示例
    install/      本地头文件、共享库和 CMake 配置
    check-build/  链接本地安装的检查程序
```

下载和构建产物由项目 `.gitignore` 忽略。安装脚本和检查源码位于 `scripts/`。
上游源码中的 README 保留原文。

## 安装和检查

在 workspace 根目录运行：

```bash
bash scripts/install_openfhe.sh
```

要求 Git、CMake >= 3.16.3、GCC >= 9、Make，以及首次下载时的网络访问。
默认 16 个构建任务，可用 `BUILD_JOBS=8 bash scripts/install_openfhe.sh` 调整。

配置为 Release、C++17、共享库、OpenMP、64 位整数后端，关闭机器特定优化。
64 位后端不是安全级别。关闭上游单元测试和 benchmarks，编译
`simple-real-numbers`、`boolean`、`scheme-switching` 三个示例。

检查使用 CKKS 的 `HEStd_128_classic` 和 FHEW 的 `STD128`，验证密文乘法
（绝对误差不超过 `1e-6`）和 AND/OR，不等于完整 scheme switching 验证。

## 在实验中使用

给 CMake 提供本地安装目录：

```bash
cmake -S path/to/experiment -B build/experiment \
  -DCMAKE_PREFIX_PATH="$PWD/thirdparty/openfhe/install"
```

使用 `find_package(OpenFHE 1.5.1 EXACT CONFIG REQUIRED)`。
可参考 `scripts/openfhe_check/CMakeLists.txt`。

上游转换示例位于：

```text
thirdparty/openfhe/build/bin/examples/pke/scheme-switching
```

部分示例使用 TOY 参数，不能直接将其计时作为正式研究基线。

参考：[固定版本](https://github.com/openfheorg/openfhe-development/releases/tag/v1.5.1)、
[Linux 安装说明](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/linux.html)。
