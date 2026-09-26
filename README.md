# CipherMoE

Research workspace for non-interactive FHE inference with private MoE routing.
The proposed SelectApply primitive takes encrypted routing and activations,
server-side plaintext expert weights, and returns an encryption of `W_e x`.
Its construction and performance benefits remain research questions.

## Layout

```text
baseline/
  Dense/              Evaluate every expert, then privately select the output
  Select_Weight/      Privately select weights, then apply them
  Routing_Publicity/  Apply the expert identified by a public index
src/                  Proposed SelectApply implementation
documents/            Research notes and initial proposal
scripts/              Dependency installation and installation checks
thirdparty/           Local external dependencies
```

See [baseline/README.md](baseline/README.md) for comparison scope and
[documents/idea.md](documents/idea.md) for the initial proposal.
The baseline and `src` directories currently contain documentation;
algorithm implementations are pending.

## OpenFHE

Install the pinned CPU build locally:

```bash
bash scripts/install_openfhe.sh
```

The install prefix is `thirdparty/openfhe/install`. Configuration and usage
are documented in [thirdparty/README.md](thirdparty/README.md).
