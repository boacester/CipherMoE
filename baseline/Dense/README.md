# Dense

Implementation pending.

For each expert, compute `Enc(W_i x)` using plaintext expert weights and
encrypted activations. Privately select the desired encrypted output using
encrypted routing information.

This baseline preserves routing privacy and evaluates every expert. It measures
whether SelectApply reduces the cost of this evaluation-and-selection strategy.
Record output selection costs as well as expert evaluation costs.
