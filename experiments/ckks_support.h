#pragma once
#include "openfhe.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ciphermoe {
using namespace lbcrypto;
using Cipher = Ciphertext<DCRTPoly>;
using Matrix = std::vector<std::vector<double>>;
using Matrices = std::vector<Matrix>;
using Diagonals = std::vector<Plaintext>;

struct Counts {
    uint64_t ct_pt = 0;
    uint64_t ct_ct = 0;
    uint64_t rotations = 0;
    uint64_t additions = 0;
    uint64_t selected_weight_peak_bytes = 0;
    uint64_t selected_weight_peak_count = 0;
};

struct Inputs {
    CryptoContext<DCRTPoly> context;
    Cipher x;
    std::vector<Cipher> selectors;
    std::vector<Diagonals> weights;
    std::vector<Diagonals> basis;
    std::vector<std::vector<Plaintext>> coefficients;
    size_t dim = 0;
    size_t public_expert = 0;
};

struct Result {
    Cipher output;
    Counts counts;
};

inline uint64_t Payload(const Cipher& cipher) {
    uint64_t bytes = 0;
    for (const auto& poly : cipher->GetElements())
        bytes += poly.GetRingDimension() * poly.GetNumOfElements() * sizeof(uint64_t);
    return bytes;
}

inline void Accumulate(const Inputs& in, Cipher& total, const Cipher& value, Counts& counts) {
    if (!total)
        total = value;
    else {
        in.context->EvalAddInPlace(total, value);
        ++counts.additions;
    }
}

inline std::vector<Cipher> RotateInput(const Inputs& in, Counts& counts) {
    std::vector<Cipher> rotations{in.x};
    for (size_t j = 1; j < in.dim; ++j) {
        rotations.push_back(in.context->EvalRotate(in.x, static_cast<int>(j)));
        ++counts.rotations;
    }
    return rotations;
}

inline Cipher Apply(const Inputs& in, const Diagonals& diagonals,
                    const std::vector<Cipher>& rotations, Counts& counts) {
    Cipher output;
    for (size_t j = 0; j < in.dim; ++j) {
        const auto term = in.context->EvalMult(rotations[j], diagonals[j]);
        ++counts.ct_pt;
        Accumulate(in, output, term, counts);
    }
    return output;
}

Result Dense(const Inputs& in);
Result SelectWeight(const Inputs& in, size_t block_size);
Result PublicRouting(const Inputs& in);
Result LowRankBasisApply(const Inputs& in);
}  // namespace ciphermoe
