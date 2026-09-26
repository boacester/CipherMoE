# Local OpenFHE Installation

OpenFHE is pinned to `v1.5.1`, commit
`1306d14f8c26bb6150d3e6ad54f28dfe1007689e`, with its upstream submodules.

```text
thirdparty/
  openfhe/
    src/          Upstream source checkout
    build/        Libraries and selected upstream examples
    install/      Local headers, shared libraries, and CMake package
    check-build/  Installation check linked against the local package
```

These downloaded and generated directories are ignored by the workspace's
`.gitignore`. The installer and check sources are in `scripts/`.

## Install and Check

From the workspace root, run:

```bash
bash scripts/install_openfhe.sh
```

Prerequisites: Git, CMake >= 3.16.3, GCC >= 9, Make, and network access for
the initial download. The script defaults to 16 build jobs; override with
`BUILD_JOBS=8 bash scripts/install_openfhe.sh`.

Build configuration: Release, C++17, shared libraries, OpenMP enabled,
64-bit native integer backend, and machine-specific optimizations disabled.
The 64-bit backend setting is an implementation choice, not a security level.
Upstream unit tests and benchmarks are disabled. The selected upstream
example targets are `simple-real-numbers`, `boolean`, and `scheme-switching`.

The installation check uses CKKS with `HEStd_128_classic` and FHEW with
`STD128`. It checks CKKS encrypted multiplication (absolute error <= 1e-6)
and encrypted AND/OR. This is an installation check, not a performance
benchmark or a full scheme-switching test.

## Use in an Experiment

Pass the local prefix to the experiment's CMake configuration:

```bash
cmake -S path/to/experiment -B build/experiment \
  -DCMAKE_PREFIX_PATH="$PWD/thirdparty/openfhe/install"
```

Use `find_package(OpenFHE 1.5.1 EXACT CONFIG REQUIRED)` in CMake. See
`scripts/openfhe_check/CMakeLists.txt` for a working consumer.

The upstream scheme-switching example is available at:

```text
thirdparty/openfhe/build/bin/examples/pke/scheme-switching
```

Some upstream examples use toy parameters. Their timings must not be used
as research baselines without configuring and recording suitable parameters.

References: [OpenFHE release](https://github.com/openfheorg/openfhe-development/releases/tag/v1.5.1)
and [Linux installation guide](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/linux.html).
