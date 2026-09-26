#include "experiments/ckks_support.h"

namespace ciphermoe {
Result BasisApply(const Inputs& in) {
    Result result;
    const auto rotations = RotateInput(in, result.counts);
    for (size_t r = 0; r < in.basis.size(); ++r) {
        Cipher coefficient;
        for (size_t expert = 0; expert < in.selectors.size(); ++expert) {
            const auto term = in.context->EvalMult(in.selectors[expert], in.coefficients[expert][r]);
            ++result.counts.ct_pt;
            Accumulate(in, coefficient, term, result.counts);
        }
        const auto basis_output = Apply(in, in.basis[r], rotations, result.counts);
        const auto selected = in.context->EvalMult(coefficient, basis_output);
        ++result.counts.ct_ct;
        Accumulate(in, result.output, selected, result.counts);
    }
    return result;
}
}  // namespace ciphermoe
