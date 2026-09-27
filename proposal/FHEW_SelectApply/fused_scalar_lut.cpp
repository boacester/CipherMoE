#include "binfhecontext.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

namespace {
constexpr uint32_t kExperts = 2;
constexpr uint32_t kActivationLevels = 2;
constexpr uint32_t kUsedDomain = kExperts * kActivationLevels;
constexpr std::array<int32_t, kExperts> kWeights{-2, 2};
constexpr std::array<int32_t, kActivationLevels> kActivations{-1, 1};

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint32_t EncodeSigned(int32_t value, uint32_t p) {
    const int32_t reduced = value % static_cast<int32_t>(p);
    return static_cast<uint32_t>(reduced < 0 ? reduced + static_cast<int32_t>(p) : reduced);
}

uint32_t FusedValue(uint32_t message, uint32_t p) {
    if (message >= kUsedDomain)
        return 0;
    const uint32_t expert = message / kActivationLevels;
    const uint32_t activation = message % kActivationLevels;
    return EncodeSigned(kWeights[expert] * kActivations[activation], p);
}

uint32_t SelectValue(uint32_t message, uint32_t) {
    // Avoid zero at LUT[0]. OpenFHE 1.5.1 checks q - LUT[q/2]
    // without modular reduction, so a mathematically negacyclic zero entry is
    // otherwise classified as an arbitrary function and costs a second PBS.
    return message < kExperts ? message + 1 : 1;
}

uint32_t MultiplyValue(uint32_t message, uint32_t p) {
    if (message >= kUsedDomain)
        return 0;
    const uint32_t weight_index = message / kActivationLevels;
    const uint32_t activation = message % kActivationLevels;
    return EncodeSigned(kWeights[weight_index] * kActivations[activation], p);
}

template <typename Function>
std::vector<NativeInteger> BuildLut(BinFHEContext& context, uint32_t p,
                                    Function function, bool negacyclic) {
    const uint32_t q = context.GetParams()->GetLWEParams()->Getq().ConvertToInt();
    if (q % p != 0)
        throw std::runtime_error("plaintext modulus must divide ciphertext modulus");
    std::vector<NativeInteger> lut(q);
    const uint32_t scale = q / p;
    for (uint32_t i = 0; i < q; ++i) {
        const uint32_t message = (static_cast<uint64_t>(i) * p) / q;
        uint32_t value;
        if (negacyclic && message >= p / 2) {
            const uint32_t positive = function(message - p / 2, p);
            value = positive == 0 ? 0 : p - positive;
        }
        else {
            value = function(message, p);
        }
        lut[i] = NativeInteger(value * scale);
    }
    return lut;
}

uint32_t InternalPbsCount(const std::vector<NativeInteger>& lut, uint32_t q) {
    const size_t midpoint = lut.size() / 2;
    // Deliberately mirror BinFHEScheme::checkInputFunction in OpenFHE 1.5.1,
    // including its non-modular q - 0 behavior.
    bool negacyclic = lut[0] == NativeInteger(q) - lut[midpoint];
    bool periodic = lut[0] == lut[midpoint];
    for (size_t i = 1; i < midpoint; ++i) {
        negacyclic = negacyclic && lut[i] == NativeInteger(q) - lut[midpoint + i];
        periodic = periodic && lut[i] == lut[midpoint + i];
    }
    return negacyclic ? 1 : (periodic ? 2 : 2);
}

LWECiphertext Joint(BinFHEContext& context, ConstLWECiphertext& high,
                    ConstLWECiphertext& low) {
    auto result = std::make_shared<LWECiphertextImpl>(*high);
    context.GetLWEScheme()->EvalMultConstEq(result, NativeInteger(kActivationLevels));
    context.GetLWEScheme()->EvalAddEq(result, low);
    return result;
}

int32_t DecodeSigned(LWEPlaintext value, uint32_t p) {
    const int32_t decoded = static_cast<int32_t>(value);
    return decoded >= static_cast<int32_t>(p / 2) ? decoded - static_cast<int32_t>(p) : decoded;
}

struct VariantResult {
    std::string name;
    std::vector<double> samples_ms;
    double median_ms = 0;
    uint32_t eval_func_calls = 0;
    uint32_t internal_pbs = 0;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(name), CEREAL_NVP(samples_ms), CEREAL_NVP(median_ms),
                CEREAL_NVP(eval_func_calls), CEREAL_NVP(internal_pbs));
    }
};
}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 3;
        if (repeats < 1)
            throw std::runtime_error("repeats must be positive");

        const auto setup_start = Clock::now();
        BinFHEContext context;
        // N=2048 gives p=8: the four legal joint inputs fit in the first half,
        // allowing a correct negacyclic extension while retaining STD128.
        context.GenerateBinFHEContext(STD128, true, 12, 2048, GINX, false);
        const auto secret_key = context.KeyGen();
        context.BTKeyGen(secret_key);
        const double key_setup_ms = Milliseconds(setup_start);
        const uint32_t p = context.GetMaxPlaintextSpace().ConvertToInt();
        const uint32_t q = context.GetParams()->GetLWEParams()->Getq().ConvertToInt();
        if (p < 2 * kUsedDomain || (p & (p - 1)) != 0)
            throw std::runtime_error("FHEW plaintext space is too small for the joint encoding");

        const auto fused_arbitrary = BuildLut(context, p, FusedValue, false);
        const auto fused_negacyclic = BuildLut(context, p, FusedValue, true);
        const auto select_arbitrary = BuildLut(context, p, SelectValue, false);
        const auto select_negacyclic = BuildLut(context, p, SelectValue, true);
        const auto multiply_arbitrary = BuildLut(context, p, MultiplyValue, false);
        const auto multiply_negacyclic = BuildLut(context, p, MultiplyValue, true);

        const auto encryption_start = Clock::now();
        std::vector<LWECiphertext> encrypted_experts;
        std::vector<LWECiphertext> encrypted_activations;
        for (uint32_t expert = 0; expert < kExperts; ++expert)
            encrypted_experts.push_back(context.Encrypt(secret_key, expert, SMALL_DIM, p));
        for (uint32_t activation = 0; activation < kActivationLevels; ++activation)
            encrypted_activations.push_back(context.Encrypt(secret_key, activation, SMALL_DIM, p));
        const double input_encryption_ms = Milliseconds(encryption_start);

        struct Variant {
            std::string name;
            const std::vector<NativeInteger>* first;
            const std::vector<NativeInteger>* second;
        };
        const std::vector<Variant> variants{
            {"fused_arbitrary", &fused_arbitrary, nullptr},
            {"select_then_apply_arbitrary", &select_arbitrary, &multiply_arbitrary},
            {"fused_negacyclic", &fused_negacyclic, nullptr},
            {"select_then_apply_negacyclic", &select_negacyclic, &multiply_negacyclic},
        };
        std::vector<VariantResult> results;
        for (const auto& variant : variants) {
            VariantResult result;
            result.name = variant.name;
            result.eval_func_calls = variant.second ? 2 : 1;
            result.internal_pbs = InternalPbsCount(*variant.first, q) +
                (variant.second ? InternalPbsCount(*variant.second, q) : 0);
            for (int trial = -1; trial < repeats; ++trial) {
                const uint32_t expert = trial < 0 ? kExperts - 1 : trial % kExperts;
                const uint32_t activation = trial < 0 ? 0 : (trial / kExperts) % kActivationLevels;
                const auto start = Clock::now();
                auto joint = Joint(context, encrypted_experts[expert], encrypted_activations[activation]);
                LWECiphertext output;
                if (!variant.second) {
                    output = context.EvalFunc(joint, *variant.first);
                }
                else {
                    auto selected = context.EvalFunc(encrypted_experts[expert], *variant.first);
                    context.GetLWEScheme()->EvalSubConstEq(selected, NativeInteger(q / p));
                    auto multiply_input = Joint(context, selected, encrypted_activations[activation]);
                    output = context.EvalFunc(multiply_input, *variant.second);
                }
                const double elapsed = Milliseconds(start);
                if (trial >= 0)
                    result.samples_ms.push_back(elapsed);
                LWEPlaintext decrypted;
                context.Decrypt(secret_key, output, &decrypted, p);
                const int32_t expected = kWeights[expert] * kActivations[activation];
                if (DecodeSigned(decrypted, p) != expected)
                    throw std::runtime_error("FHEW lookup result differs from integer reference");
            }
            auto sorted = result.samples_ms;
            std::sort(sorted.begin(), sorted.end());
            result.median_ms = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
            results.push_back(result);
        }

        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        const uint64_t peak_rss_kib = usage.ru_maxrss;
        const uint32_t ring_dim = context.GetParams()->GetLWEParams()->GetN();
        cereal::JSONOutputArchive output(std::cout);
        output(CEREAL_NVP(results), CEREAL_NVP(key_setup_ms), CEREAL_NVP(input_encryption_ms),
               CEREAL_NVP(p), CEREAL_NVP(q), CEREAL_NVP(ring_dim), CEREAL_NVP(peak_rss_kib));
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
