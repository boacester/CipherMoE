# SelectApply

This directory is reserved for the proposed research implementation.
No concrete construction is implemented yet.

Target interface:

```text
SelectApply(Enc(e), Enc(x), {W_i}) -> Enc(W_e x)
```

The construction should protect routing without intermediate client interaction.
Its goal is to fuse private selection with operator application and reduce
online costs. Avoiding a full encrypted weight allocation alone does not
establish a reduction in cryptographic computation.

Compare a concrete construction against `../baseline/`. Quantization and model
preprocessing are allowed, with their errors and costs explicitly recorded.
