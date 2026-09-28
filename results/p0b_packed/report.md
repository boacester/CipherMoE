# P0b: packed selector reuse (2026-09-27)

## Scope and implementation

Test `PackedSelect(Enc(e), public {V_i}) -> Enc(V_e)`, with arbitrary,
independently chosen unsigned 4-bit values in each public vector. There is no
activation, matrix multiplication, CKKS conversion, or low-rank assumption.
Materializing `m` output LWEs is allowed **only in this selector diagnostic**;
it would violate the final Select-and-Apply design requirement for whole weights.

Reuse TFHE-rs commit `079f46d08ffb8763de54b7794e850ddbd98fe63b`
(BSD-3-Clause-Clear): `shortint::ServerKey::generate_many_lookup_table` creates
the accumulator container; `apply_many_lookup_table` performs **one key switch,
one blind rotation and multiple sample extractions per accumulator** in this
parameter's standard key-switch/bootstrap order. Instead of separate functions
or sub-LUTs, `p0b_packed.rs` puts the public vector of expert `i` into box `i`
of one GLWE polynomial, and sets the coefficient-extraction stride explicitly.
The data, access pattern, tile count and loop bounds do not depend on secret `e`.

Fixed key parameter: `V1_8_PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128`,
polynomial size 8192, native `u64` torus, output modulus `8*8=64` and output
values 0..15. The parameter name corresponds to a reported failure probability
of about `2^-128`, **not an independently audited 128-bit security claim**.
The selector uses an `N`-value torus encoding, so one expert gets `8192/N`
coefficients: 4096 / 1024 / 128 for `N=2/8/64`. TFHE-rs' public
`encrypt_with_message_and_carry_modulus` still encrypts using the key's original
64-value torus encoding even with smaller metadata: the *client*, which knows
`e`, adjusts the LWE body from `e*2^57` to `e*2^63/N` before submission.
It also applies the fixed public coefficient-centering offset. The server
neither needs `e` nor performs an `e`-dependent correction. The GLWE output is
encoded for 64 values, so its extracted shortint output metadata is changed to
message=8, carry=8 before decryption. This is a pinned-version experimental
adaptation, not a stock shortint API guarantee. A first scan that changed only
message/carry metadata without changing the torus phase was **invalid and was
discarded**; all `raw/` benchmark JSONL uses the corrected client encoding.
An additional explicit, non-hash public vector table (`experiments/p0_custom_vectors.json`)
at `N=2,m=8` passed **32/32** encrypted-output checks.

## Measurements

All 21 configurations use `N={2,8,64}`, `m={1,8,32,64,128,256,512}`.
For the guarded layout, `(stride, margin, outputs/tile)` is `(32,32,126)`
for `N=2`, `(32,32,30)` for `N=8`, `(8,16,12)` for `N=64`.
CPU: Intel Xeon Platinum 8592V. `RAYON_NUM_THREADS` was not overridden for
these runs, unlike the previous P0 report; comparisons with P0 are therefore
indicative, not controlled same-thread speedups. A key is generated once for
each `N` and reused across the seven `m` values. Online latency is the median
of **two** timed trials after the per-expert correctness trials; client
encryption, accumulator construction and decryption are excluded. Every trial
uses a fresh encrypted selector. For `N=2/8`, verify **every expert** plus two
timed trials; for `N=64`, verify IDs `0,1,16,32,62,63` plus two timed trials,
not all 64 IDs. Every output in those trials is decrypted and compared with
the corresponding public value. Times rounded to milliseconds.

| m | N=2: BR / ms / wrong | N=8: BR / ms / wrong | N=64: BR / ms / wrong |
| ---: | ---: | ---: | ---: |
| 1 | 1 / 227 / 0 | 1 / 198 / 0 | 1 / 225 / 5 |
| 8 | 1 / 222 / 0 | 1 / 217 / 0 | 1 / 198 / 39 |
| 32 | 1 / 218 / 0 | 2 / 425 / 0 | 3 / 621 / 148 |
| 64 | 1 / 230 / 0 | 3 / 620 / 0 | 6 / 1216 / 363 |
| 128 | 2 / 444 / 0 | 5 / 1019 / 0 | 11 / 2249 / 121 |
| 256 | 3 / 653 / 0 | 9 / 1822 / 0 | 22 / 4441 / 709 |
| 512 | **5 / 1063 / 0** | **18 / 3565 / 0** | **43 / 8486 / 2391** |

At `m=512`, actual BR/output is respectively **0.00977 / 0.03516 /
0.08398**, and server latency/output is **2.08 / 6.96 / 16.57 ms**.
The last ratio is **incorrect computation**: 2391 of 4096 checked output
ciphertexts fail. Dense packing without any guard uses only **1/1/4 BR** for
`N=2/8/64,m=512`, but produces **1907/4812/3362 errors** in respectively
2048/5120/4096 checks. Thus the nominal 4-BR result at 64 experts is not a
viable selector. Dense runs for all 21 sizes are in `raw/dense_n*.jsonl`.
The guarded `N=2/8,m=512` tests passed 2048/2048 and 5120/5120 output
checks, respectively; this is finite experimental evidence, not a proof of
negligible error probability for all future encryptions.

To isolate the failure mechanism, a 60-encryption probe sampled the selected
coefficient's 8-coefficient bins for three IDs at `N=64`. The decoded location
varied across bins 6..9 around nominal bin 8, rather than staying at one
coefficient; see `raw/phase_n64.jsonl`. Repeating an arbitrary output in only
one or eight consecutive coefficients cannot reliably survive this phase
variation. This is an **empirical observation**, not a worst-case noise bound.
With a wider `stride=32,margin=32`, a `N=64` tile fits **2 outputs**: the
short scan `m=1/8/32` measured 1/4/16 BR, and 0 failures in 8/64/256 checked
outputs respectively. A 512-output extension would require **256 BR by
capacity**, not a measured 512-output latency or proven zero error rate.

## Resources and limits

At `m=512`, stored accumulator memory for each model configuration is 655360 / 2359296 /
5636096 bytes for guarded `N=2/8/64`; each GLWE accumulator is 131072 bytes.
All three use the same serialized evaluation key size **989461560 bytes**;
one request ciphertext is **65616 bytes**, and 512 separate output LWE
ciphertexts total **33595392 bytes**. Public vector storage is `N*m` bytes
in this prototype. Accumulator preparation takes approximately 0.16 / 0.88 /
2.34 ms at `m=512`, respectively; key generation about 3.3 s per process.
An isolated call to the same sample-extraction primitive, on an unrotated
GLWE of matching shape, took about **3.6 / 3.6 / 3.4 ms for 512 outputs**.
This is an extraction *proxy*, not an instrumented component timing inside
online PBS; GLWE/LWE cloning and memory allocation also contribute online.
There are `m` LWE extractions and `ceil(m/outputs_per_tile)` key switches and
blind rotations; the result is **not** a packed ciphertext directly usable as
CKKS slots. No timing here measures dot products, matrix tiles, scheme
switching, or hidden server memory-access properties.

## Go / no-go

**Go for small-`N` packed selection only:** `N=2,m=512` achieves 5 BR and
`N=8,m=512` achieves 18 BR with zero errors in the stated finite tests.
**No-go for the requested 64-expert 1–8 BR target under this parameter and
coefficient-packing method:** 4 BR (dense) and 43 BR (guarded) both fail
correctness; the measured 2-outputs/BR wide layout cannot meet the target.
Do **not** extend this method to dot product or matrix tiles on the basis of
its incorrect low-BR runs. This does not rule out GLWE CMux/vertical packing,
alternative noise-control mechanisms, or a compact encrypted-control object
for CKKS; none was benchmarked here. Any next candidate must count scheme
conversion, plaintext-model encoding, encrypted-control cost and full output
materialization rather than transferring those costs off the ledger.

## Reproduction

```bash
RUSTUP_TOOLCHAIN=1.91.1 CARGO_TARGET_DIR="$PWD/build/tfhe-target" \
  cargo build --release --offline --manifest-path experiments/tfhe_stage_b/Cargo.toml --bin p0b_packed
for n in 2 8 64; do
  build/tfhe-target/release/p0b_packed "$n" 1 2 0
done
build/tfhe-target/release/p0b_packed 2 32 2 32
build/tfhe-target/release/p0b_packed 8 32 2 32
build/tfhe-target/release/p0b_packed 64 8 2 16
build/tfhe-target/release/p0b_packed 64 32 2 32 32
build/tfhe-target/release/p0b_packed probe 64 20
build/tfhe-target/release/p0b_packed 2 32 2 32 512 experiments/p0_custom_vectors.json
```

Each scan prints one JSON object per `m`. The original measurements are in
`raw/{dense,guarded}_n{2,8,64}.jsonl`, `raw/wide_n64_to32.jsonl` and
`raw/phase_n64.jsonl` and `raw/custom_n2_m8.jsonl`.
