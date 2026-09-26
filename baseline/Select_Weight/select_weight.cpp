#include "experiments/ckks_support.h"

namespace ciphermoe {
Result SelectWeight(const Inputs& in, size_t block_size) {
    Result result;
    const auto rotations = RotateInput(in, result.counts);
    for (size_t start = 0; start < in.dim; start += block_size) {
        std::vector<Cipher> selected;
        uint64_t live_bytes = 0;
        const size_t end = std::min(in.dim, start + block_size);
        for (size_t j = start; j < end; ++j) {
            Cipher diagonal;
            for (size_t expert = 0; expert < in.weights.size(); ++expert) {
                const auto term = in.context->EvalMult(in.selectors[expert], in.weights[expert][j]);
                ++result.counts.ct_pt;
                Accumulate(in, diagonal, term, result.counts);
            }
            live_bytes += Payload(diagonal);
            selected.push_back(diagonal);
        }
        result.counts.selected_weight_peak_bytes =
            std::max(result.counts.selected_weight_peak_bytes, live_bytes);
        result.counts.selected_weight_peak_count =
            std::max<uint64_t>(result.counts.selected_weight_peak_count, selected.size());
        for (size_t j = start; j < end; ++j) {
            const auto term = in.context->EvalMult(selected[j - start], rotations[j]);
            ++result.counts.ct_ct;
            Accumulate(in, result.output, term, result.counts);
        }
    }
    return result;
}
}  // namespace ciphermoe
