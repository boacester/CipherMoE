#include "experiments/ckks_support.h"

namespace ciphermoe {
Result PublicRouting(const Inputs& in) {
    Result result;
    const auto rotations = RotateInput(in, result.counts);
    result.output = Apply(in, in.weights.at(in.public_expert), rotations, result.counts);
    return result;
}
}  // namespace ciphermoe
