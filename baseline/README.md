# Linear Primitive Baselines

The initial comparison computes `y = W_e x` for a single token and one selected
expert. All methods use the same inputs, expert matrices, and numerical
reference. Private methods receive encrypted activations and routing information.
`Routing_Publicity` receives a plaintext expert index and encrypted activations.

| Directory | Method | Comparison purpose |
| --- | --- | --- |
| `Dense/` | Evaluate all plaintext-weight operators on encrypted activations, then privately select | Cost of evaluating every expert |
| `Select_Weight/` | Privately select encrypted weights, then apply them to encrypted activations | Cost of weight selection and ciphertext-ciphertext multiplication |
| `Routing_Publicity/` | Apply the plaintext-weight operator named by a public index | Performance reference when routing is known to the server |

The proposed method belongs in `../src/`. Implementations are pending.

## Comparison Contract

- Match numerical ranges, precision requirements, and target security level.
- Report the routing representation and include any required index-to-selector
  conversion in the primitive total, or explicitly report it as an excluded cost.
- Separate model preprocessing and client key setup from online evaluation.
- Report online latency, error, peak memory, and expensive operation counts.
- Record packing, cryptographic parameters, thread count, and hardware.
- Treat router score computation and top-k generation as later integration work;
  the initial experiment starts from an already supplied routing choice.

## Optional Materialization Ablation

`Select_Weight/` can also contain a streaming variant: select one weight block
or encoded diagonal, apply it immediately, and accumulate its contribution.
This reduces simultaneous intermediate storage while retaining encrypted
selected blocks and their ciphertext-ciphertext products.

Compare the materialized and streaming versions to measure the effect of
intermediate storage alone. This is an optional ablation, not a prerequisite
for implementing SelectApply. The main research comparison is between a concrete
SelectApply construction in `src/` and the baselines above.
