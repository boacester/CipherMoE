#include "experiments/ckks_support.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/vector.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

using namespace ciphermoe;
using Clock = std::chrono::steady_clock;

static double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static Diagonals Encode(CryptoContext<DCRTPoly> context, const ciphermoe::Matrix& matrix) {
    const size_t dim = matrix.size();
    Diagonals diagonals;
    for (size_t j = 0; j < dim; ++j) {
        std::vector<double> values(dim);
        for (size_t row = 0; row < dim; ++row)
            values[row] = matrix.at(row).at((row + j) % dim);
        diagonals.push_back(context->MakeCKKSPackedPlaintext(values));
    }
    return diagonals;
}

static std::vector<double> Reference(const ciphermoe::Matrix& matrix, const std::vector<double>& x) {
    std::vector<double> output(matrix.size(), 0.0);
    for (size_t row = 0; row < matrix.size(); ++row)
        for (size_t col = 0; col < x.size(); ++col)
            output[row] += matrix.at(row).at(col) * x[col];
    return output;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4)
            throw std::runtime_error("usage: matrix_benchmark MODEL.json METHOD REPEATS [BLOCK]");
        const std::string method = argv[2];
        const int repeats = std::stoi(argv[3]);
        if (repeats < 1)
            throw std::runtime_error("repeats must be positive");
        Matrices weights, basis, approximation;
        ciphermoe::Matrix coefficients, inputs;
        std::ifstream file(argv[1]);
        if (!file)
            throw std::runtime_error("cannot open model");
        {
            cereal::JSONInputArchive archive(file);
            archive(cereal::make_nvp("weights", weights), cereal::make_nvp("basis", basis),
                    cereal::make_nvp("approximation", approximation),
                    cereal::make_nvp("coefficients", coefficients), cereal::make_nvp("inputs", inputs));
        }
        const size_t n = weights.size();
        const size_t dim = weights.at(0).size();
        const size_t rank = basis.size();
        const size_t block_size = argc > 4 ? std::stoul(argv[4]) : dim;
        if (block_size < 1 || inputs.size() < 3)
            throw std::runtime_error("invalid block size or inputs");

        const auto key_start = Clock::now();
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(3);
        parameters.SetScalingModSize(45);
        parameters.SetFirstModSize(60);
        parameters.SetBatchSize(dim);
        parameters.SetSecurityLevel(HEStd_128_classic);
        Inputs prepared;
        prepared.context = GenCryptoContext(parameters);
        prepared.context->Enable(PKE);
        prepared.context->Enable(KEYSWITCH);
        prepared.context->Enable(LEVELEDSHE);
        prepared.context->Enable(ADVANCEDSHE);
        prepared.dim = dim;
        const auto keys = prepared.context->KeyGen();
        prepared.context->EvalMultKeyGen(keys.secretKey);
        std::vector<int> rotation_indices;
        for (size_t j = 1; j < dim; ++j)
            rotation_indices.push_back(static_cast<int>(j));
        prepared.context->EvalRotateKeyGen(keys.secretKey, rotation_indices);
        const double key_setup_ms = Milliseconds(key_start);

        const auto encoding_start = Clock::now();
        if (method == "basis") {
            for (const auto& matrix : basis)
                prepared.basis.push_back(Encode(prepared.context, matrix));
            for (const auto& row : coefficients) {
                std::vector<Plaintext> encoded;
                for (double value : row)
                    encoded.push_back(prepared.context->MakeCKKSPackedPlaintext(std::vector<double>(dim, value)));
                prepared.coefficients.push_back(encoded);
            }
        }
        else {
            for (const auto& matrix : weights)
                prepared.weights.push_back(Encode(prepared.context, matrix));
        }
        const double encoding_ms = Milliseconds(encoding_start);

        const auto encryption_start = Clock::now();
        std::vector<Cipher> encrypted_inputs;
        for (const auto& x : inputs)
            encrypted_inputs.push_back(prepared.context->Encrypt(keys.publicKey,
                prepared.context->MakeCKKSPackedPlaintext(x)));
        std::vector<std::vector<Cipher>> selectors_by_route;
        for (size_t route = 0; route < n; ++route) {
            std::vector<Cipher> selectors;
            for (size_t expert = 0; expert < n; ++expert) {
                const auto plaintext = prepared.context->MakeCKKSPackedPlaintext(
                    std::vector<double>(dim, expert == route ? 1.0 : 0.0));
                selectors.push_back(prepared.context->Encrypt(keys.publicKey, plaintext));
            }
            selectors_by_route.push_back(selectors);
        }
        const double input_encryption_ms = Milliseconds(encryption_start);

        auto evaluate = [&]() {
            if (method == "dense") return Dense(prepared);
            if (method == "select_weight") return SelectWeight(prepared, block_size);
            if (method == "public") return PublicRouting(prepared);
            if (method == "basis") return BasisApply(prepared);
            throw std::runtime_error("unknown method");
        };
        std::vector<double> samples;
        Counts counts;
        double max_he_error = 0.0, max_total_error = 0.0;
        double reference_energy = 0.0, approximation_energy = 0.0;
        // Two unmeasured cases exercise zero and signed boundary inputs.
        for (int trial = -2; trial < repeats; ++trial) {
            const size_t route = trial < 0 ? n - 1 : static_cast<size_t>(trial) % n;
            const size_t input_index = trial == -2 ? 1 : trial == -1 ? 2 :
                (static_cast<size_t>(trial) / n) % inputs.size();
            prepared.public_expert = route;
            prepared.x = encrypted_inputs.at(input_index);
            prepared.selectors = selectors_by_route.at(route);
            const auto start = Clock::now();
            const auto result = evaluate();
            const double elapsed = Milliseconds(start);
            if (trial >= 0)
                samples.push_back(elapsed);
            counts = result.counts;
            Plaintext decrypted;
            prepared.context->Decrypt(keys.secretKey, result.output, &decrypted);
            decrypted->SetLength(dim);
            const auto actual = decrypted->GetCKKSPackedValue();
            const auto full = Reference(weights.at(route), inputs.at(input_index));
            const auto target = method == "basis" ?
                Reference(approximation.at(route), inputs.at(input_index)) : full;
            for (size_t j = 0; j < dim; ++j) {
                const double error = std::abs(actual.at(j) - target[j]);
                if (!std::isfinite(error) || error > 1e-5)
                    throw std::runtime_error("encrypted result differs from plaintext reference");
                max_he_error = std::max(max_he_error, error);
                max_total_error = std::max(max_total_error, std::abs(actual.at(j) - full[j]));
                reference_energy += full[j] * full[j];
                approximation_energy += (target[j] - full[j]) * (target[j] - full[j]);
            }
        }
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double median_ms = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        const uint64_t peak_rss_kib = usage.ru_maxrss;
        const uint64_t ring_dim = prepared.context->GetRingDimension();
        const double relative_output_error = std::sqrt(approximation_energy / std::max(reference_energy, 1e-30));
        cereal::JSONOutputArchive output(std::cout);
        output(cereal::make_nvp("method", method), cereal::make_nvp("n", n),
               cereal::make_nvp("dim", dim), cereal::make_nvp("rank", rank),
               cereal::make_nvp("block_size", block_size), cereal::make_nvp("ring_dim", ring_dim),
               cereal::make_nvp("samples_ms", samples), cereal::make_nvp("median_ms", median_ms),
               cereal::make_nvp("key_setup_ms", key_setup_ms), cereal::make_nvp("encoding_ms", encoding_ms),
               cereal::make_nvp("input_encryption_ms", input_encryption_ms),
               cereal::make_nvp("max_he_error", max_he_error), cereal::make_nvp("max_total_error", max_total_error),
               cereal::make_nvp("relative_output_error", relative_output_error),
               cereal::make_nvp("ct_pt", counts.ct_pt), cereal::make_nvp("ct_ct", counts.ct_ct),
               cereal::make_nvp("rotations", counts.rotations), cereal::make_nvp("additions", counts.additions),
               cereal::make_nvp("selected_weight_peak_bytes", counts.selected_weight_peak_bytes),
               cereal::make_nvp("selected_weight_peak_count", counts.selected_weight_peak_count),
               cereal::make_nvp("peak_rss_kib", peak_rss_kib));
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
