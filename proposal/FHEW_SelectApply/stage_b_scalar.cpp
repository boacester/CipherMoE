#include "binfhecontext.h"
#include "utils/serial.h"
#include "cereal/archives/json.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int kXLevels = 4;
constexpr int kOutputModulus = 16;

double Ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int X(int code) { return code - 2; }
int Encoded(int value, int p) { return (value % p + p) % p; }

int Decode(int value) {
    value %= kOutputModulus;
    return value >= kOutputModulus / 2 ? value - kOutputModulus : value;
}

std::vector<NativeInteger> MakeLut(const std::vector<int>& weights, int p, int q, bool negacyclic) {
    const int n = weights.size();
    std::vector<NativeInteger> lut(q);
    for (int i = 0; i < q; ++i) {
        int code = i / (q / p);
        int value = 0;
        if (negacyclic && code >= p / 2) {
            const int lower = code - p / 2;
            value = -((lower / kXLevels < n)
                          ? weights[lower / kXLevels] * X(lower % kXLevels) + 5
                          : 1);
        }
        else if (code / kXLevels < n) {
            value = weights[code / kXLevels] * X(code % kXLevels) + (negacyclic ? 5 : 0);
        }
        else if (negacyclic) {
            value = 1;
        }
        lut[i] = NativeInteger(Encoded(value, p) * (q / p));
    }
    return lut;
}

bool IsNegacyclic(const std::vector<NativeInteger>& lut, int q) {
    for (int i = 0; i < q / 2; ++i) {
        if (lut[i] != NativeInteger(q) - lut[i + q / 2])
            return false;
    }
    return true;
}

struct Stats {
    int n;
    std::string method;
    int p;
    int q;
    int ring_dim;
    int id_bits;
    int x_bits = 2;
    int output_modulus = kOutputModulus;
    int lut_entries = 0;
    uint64_t lut_bytes = 0;
    uint64_t evaluation_key_bytes = 0;
    double keygen_ms = 0;
    double preprocessing_ms = 0;
    double encryption_ms = 0;
    double decryption_ms = 0;
    std::vector<double> online_ms;
    double median_ms = 0;
    uint64_t peak_rss_kib = 0;
    int verified = 0;
    int failures = 0;
    int pbs_per_eval = 0;
    int cmux_calls_per_eval = 0;

    template <class Archive> void serialize(Archive& archive) {
        archive(CEREAL_NVP(n), CEREAL_NVP(method), CEREAL_NVP(p), CEREAL_NVP(q),
                CEREAL_NVP(ring_dim), CEREAL_NVP(id_bits), CEREAL_NVP(x_bits),
                CEREAL_NVP(output_modulus), CEREAL_NVP(lut_entries), CEREAL_NVP(lut_bytes),
                CEREAL_NVP(evaluation_key_bytes), CEREAL_NVP(keygen_ms),
                CEREAL_NVP(preprocessing_ms), CEREAL_NVP(encryption_ms),
                CEREAL_NVP(decryption_ms),
                CEREAL_NVP(online_ms), CEREAL_NVP(median_ms), CEREAL_NVP(peak_rss_kib),
                CEREAL_NVP(verified), CEREAL_NVP(failures), CEREAL_NVP(pbs_per_eval),
                CEREAL_NVP(cmux_calls_per_eval));
    }
};

// Public constants are trivial LWE ciphertexts, derived without the client's secret key.
LWECiphertext PublicBit(BinFHEContext& cc, ConstLWECiphertext& input, int bit) {
    auto ct = std::make_shared<LWECiphertextImpl>(*input);
    cc.GetLWEScheme()->EvalSubEq(ct, input);
    if (bit)
        cc.GetLWEScheme()->EvalAddConstEq(ct, NativeInteger(ct->GetModulus() / 4));
    return ct;
}

struct BooleanEval {
    BinFHEContext& cc;
    int pbs = 0;
    int cmux = 0;
    LWECiphertext zero;
    LWECiphertext one;
    std::map<int, LWECiphertext> functions;
    std::map<std::tuple<LWECiphertext, LWECiphertext, LWECiphertext>, LWECiphertext> mux_cache;

    BooleanEval(BinFHEContext& context, const std::vector<LWECiphertext>& x)
        : cc(context), zero(PublicBit(context, x[0], 0)), one(PublicBit(context, x[0], 1)) {
        functions[0] = zero;
        functions[15] = one;
        functions[10] = x[0];
        functions[12] = x[1];
        functions[5] = cc.EvalNOT(x[0]);
        functions[3] = cc.EvalNOT(x[1]);
    }

    LWECiphertext Mux(const LWECiphertext& low, const LWECiphertext& high,
                      const LWECiphertext& selector) {
        if (low == high)
            return low;
        auto key = std::make_tuple(low, high, selector);
        auto found = mux_cache.find(key);
        if (found != mux_cache.end())
            return found->second;
        ++cmux;
        pbs += 3;  // OpenFHE's CMUX is three NAND bootstraps.
        auto result = cc.EvalBinGate(CMUX, std::vector<LWECiphertext>{low, high, selector});
        mux_cache[key] = result;
        return result;
    }

    LWECiphertext Function(int mask, const std::vector<LWECiphertext>& x) {
        auto found = functions.find(mask);
        if (found != functions.end())
            return found->second;
        // Shannon expansion in x[1], using x[0] and its complement as leaves.
        const int low = mask & 3;
        const int high = (mask >> 2) & 3;
        auto digit = [&](int bits) -> LWECiphertext {
            if (bits == 0) return zero;
            if (bits == 3) return one;
            return functions.at(bits == 2 ? 10 : 5);
        };
        auto result = Mux(digit(low), digit(high), x[1]);
        functions[mask] = result;
        return result;
    }
};

std::vector<LWECiphertext> BoolApply(BooleanEval& eval, const std::vector<LWECiphertext>& id,
                                     const std::vector<LWECiphertext>& x,
                                     const std::vector<int>& weights) {
    const int n = weights.size();
    std::vector<LWECiphertext> output;
    for (int bit = 0; bit < 4; ++bit) {
        std::vector<LWECiphertext> nodes;
        for (int e = 0; e < n; ++e) {
            int mask = 0;
            for (int code = 0; code < kXLevels; ++code)
                mask |= ((Encoded(weights[e] * X(code), kOutputModulus) >> bit) & 1) << code;
            nodes.push_back(eval.Function(mask, x));
        }
        for (size_t level = 0; level < id.size(); ++level) {
            std::vector<LWECiphertext> next;
            for (size_t i = 0; i < nodes.size(); i += 2)
                next.push_back(eval.Mux(nodes[i], nodes[i + 1], id[level]));
            nodes = std::move(next);
        }
        output.push_back(nodes[0]);
    }
    return output;
}

uint64_t SerializedKeyBytes(const BinFHEContext& cc) {
    class CountingBuffer : public std::streambuf {
    public:
        uint64_t bytes = 0;
    protected:
        std::streamsize xsputn(const char*, std::streamsize count) override {
            bytes += count;
            return count;
        }
        int overflow(int ch) override {
            if (ch != traits_type::eof()) ++bytes;
            return ch;
        }
    } buffer;
    std::ostream sink(&buffer);
    Serial::Serialize(cc.GetRefreshKey(), sink, SerType::BINARY);
    Serial::Serialize(cc.GetSwitchKey(), sink, SerType::BINARY);
    return buffer.bytes;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6)
            throw std::runtime_error("usage: stage_b_scalar N fused|negacyclic|cmux checks repeats [comma-separated-weights]");
        const int n = std::stoi(argv[1]);
        const std::string method = argv[2];
        const int checks = std::stoi(argv[3]);
        const int repeats = std::stoi(argv[4]);
        if (n < 2 || n > 64 || (n & (n - 1)) || checks < 1 || repeats < 1 ||
            (method != "fused" && method != "negacyclic" && method != "cmux"))
            throw std::runtime_error("invalid parameters");
        std::vector<int> weights;
        if (argc == 6) {
            std::stringstream stream(argv[5]);
            std::string token;
            while (std::getline(stream, token, ',')) {
                size_t end = 0;
                const int value = std::stoi(token, &end);
                if (end != token.size() || value < -2 || value > 2)
                    throw std::runtime_error("weights must be integers in [-2,2]");
                weights.push_back(value);
            }
        }
        else {
            for (int e = 0; e < n; ++e)
                weights.push_back(((e * 7 + 3) % 5) - 2);
        }
        if (static_cast<int>(weights.size()) != n)
            throw std::runtime_error("weight table must have exactly N elements");
        const int p = method == "cmux" ? 4 : std::max(16, n * kXLevels * (method == "negacyclic" ? 2 : 1));
        const int ring = method == "cmux" ? 2048 : p * 256;
        BinFHEContext cc;
        cc.GenerateBinFHEContext(STD128, true, 12, ring, GINX, false);
        auto key_start = Clock::now();
        const auto sk = cc.KeyGen();
        cc.BTKeyGen(sk);
        Stats stats;
        stats.n = n;
        stats.method = method;
        stats.p = p;
        stats.q = cc.GetParams()->GetLWEParams()->Getq().ConvertToInt();
        stats.ring_dim = cc.GetParams()->GetLWEParams()->GetN();
        stats.id_bits = 0;
        for (int v = n; v > 1; v >>= 1) ++stats.id_bits;
        stats.keygen_ms = Ms(key_start);
        if (stats.q / p < 256)
            throw std::runtime_error("insufficient plaintext noise margin");
        auto prep_start = Clock::now();
        std::vector<NativeInteger> lut;
        if (method != "cmux") {
            lut = MakeLut(weights, p, stats.q, method == "negacyclic");
            if (method == "negacyclic" && !IsNegacyclic(lut, stats.q))
                throw std::runtime_error("constructed LUT is not negacyclic");
            stats.lut_entries = lut.size();
            stats.lut_bytes = lut.size() * sizeof(NativeInteger);
        }
        stats.preprocessing_ms = Ms(prep_start);
        stats.evaluation_key_bytes = SerializedKeyBytes(cc);

        // Encrypt every legal input once; online evaluation never accesses the secret key.
        auto enc_start = Clock::now();
        std::vector<LWECiphertext> ids, xs;
        if (method != "cmux") {
            for (int e = 0; e < n; ++e)
                ids.push_back(cc.Encrypt(sk, e, SMALL_DIM, p));
            for (int code = 0; code < kXLevels; ++code)
                xs.push_back(cc.Encrypt(sk, code, SMALL_DIM, p));
        }
        stats.encryption_ms = Ms(enc_start);

        for (int trial = 0; trial < checks + repeats; ++trial) {
            int e = trial < checks
                ? (checks >= n * kXLevels ? (trial / kXLevels) % n
                                           : (trial * (n - 1)) / std::max(1, checks - 1))
                : (trial * 17 + 3) % n;
            int code = trial < checks ? trial % kXLevels : (trial + 1) % kXLevels;
            if (method == "cmux") {
                auto input_start = Clock::now();
                ids.clear();
                xs.clear();
                for (int bit = 0; bit < stats.id_bits; ++bit)
                    ids.push_back(cc.Encrypt(sk, (e >> bit) & 1, SMALL_DIM, 4));
                for (int bit = 0; bit < 2; ++bit)
                    xs.push_back(cc.Encrypt(sk, (code >> bit) & 1, SMALL_DIM, 4));
                stats.encryption_ms += Ms(input_start);
            }
            auto start = Clock::now();
            int result = 0;
            if (method == "cmux") {
                BooleanEval boolean{cc, xs};
                auto output = BoolApply(boolean, ids, xs, weights);
                stats.pbs_per_eval = boolean.pbs;
                stats.cmux_calls_per_eval = boolean.cmux;
                const double elapsed = Ms(start);
                auto decrypt_start = Clock::now();
                for (int bit = 0; bit < 4; ++bit) {
                    LWEPlaintext plain;
                    cc.Decrypt(sk, output[bit], &plain, 4);
                    result |= (static_cast<int>(plain) & 1) << bit;
                }
                result = Decode(result);
                stats.decryption_ms += Ms(decrypt_start);
                if (trial >= checks) stats.online_ms.push_back(elapsed);
            }
            else {
                auto joint = std::make_shared<LWECiphertextImpl>(*ids[e]);
                cc.GetLWEScheme()->EvalMultConstEq(joint, NativeInteger(kXLevels));
                cc.GetLWEScheme()->EvalAddEq(joint, xs[code]);
                auto output = cc.EvalFunc(joint, lut);
                const double elapsed = Ms(start);
                auto decrypt_start = Clock::now();
                LWEPlaintext plain;
                cc.Decrypt(sk, output, &plain, p);
                stats.decryption_ms += Ms(decrypt_start);
                result = Decode(static_cast<int>(plain) - (method == "negacyclic" ? 5 : 0));
                stats.pbs_per_eval = IsNegacyclic(lut, stats.q) ? 1 : 2;
                if (trial >= checks) stats.online_ms.push_back(elapsed);
            }
            ++stats.verified;
            if (result != weights[e] * X(code)) ++stats.failures;
        }
        std::sort(stats.online_ms.begin(), stats.online_ms.end());
        stats.median_ms = (stats.online_ms[(repeats - 1) / 2] + stats.online_ms[repeats / 2]) / 2;
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        stats.peak_rss_kib = usage.ru_maxrss;
        cereal::JSONOutputArchive output(std::cout);
        output(cereal::make_nvp("result", stats));
        return stats.failures ? 2 : 0;
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
