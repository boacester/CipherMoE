#include "experiments/ckks_support.h"

namespace ciphermoe {
Result Dense(const Inputs& in) {
    Result result;
    const auto rotations = RotateInput(in, result.counts);
    for (size_t expert = 0; expert < in.weights.size(); ++expert) {
        const auto expert_output = Apply(in, in.weights[expert], rotations, result.counts);
        const auto selected = in.context->EvalMult(expert_output, in.selectors[expert]);
        ++result.counts.ct_ct;
        Accumulate(in, result.output, selected, result.counts);
    }
    return result;
}
}  // namespace ciphermoe
