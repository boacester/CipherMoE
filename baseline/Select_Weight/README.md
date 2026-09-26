# Select_Weight

Implementation pending.

Privately select the expert's encoded weight matrix or diagonals into encrypted
weights, then apply them to encrypted activations. Include both selection and
ciphertext-ciphertext application costs.

An optional streaming variant selects and applies one block or diagonal at a
time. It tests the memory and latency effects of avoiding full simultaneous
materialization. It still forms encrypted selected blocks and performs
ciphertext-ciphertext products.

This variant is an ablation within Select_Weight; it does not establish the
proposed SelectApply construction.
