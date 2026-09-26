#include "openfhe.h"
#include "binfhecontext.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace lbcrypto;
    try {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(1);
        parameters.SetScalingModSize(50);
        parameters.SetSecurityLevel(HEStd_128_classic);
        const auto context = GenCryptoContext(parameters);
        context->Enable(PKE);
        context->Enable(KEYSWITCH);
        context->Enable(LEVELEDSHE);
        const auto keys = context->KeyGen();
        context->EvalMultKeyGen(keys.secretKey);
        const auto input = context->MakeCKKSPackedPlaintext(std::vector<double>{1.25, -2.0, 0.5});
        const auto encrypted = context->Encrypt(keys.publicKey, input);
        const auto squared = context->EvalMult(encrypted, encrypted);
        Plaintext decrypted;
        context->Decrypt(keys.secretKey, squared, &decrypted);
        decrypted->SetLength(3);
        const auto values = decrypted->GetCKKSPackedValue();
        const std::vector<double> expected{1.5625, 4.0, 0.25};
        for (size_t i = 0; i < expected.size(); ++i) {
            const double error = std::abs(values.at(i) - expected[i]);
            if (!std::isfinite(error) || error > 1e-6)
                throw std::runtime_error("CKKS multiplication check failed");
        }
        std::cout << "CKKS multiplication passed (absolute error <= 1e-6)." << std::endl;

        BinFHEContext booleanContext;
        booleanContext.GenerateBinFHEContext(STD128);
        const auto secret = booleanContext.KeyGen();
        booleanContext.BTKeyGen(secret);
        const auto zero = booleanContext.Encrypt(secret, 0);
        const auto one = booleanContext.Encrypt(secret, 1);
        LWEPlaintext result;
        booleanContext.Decrypt(secret, booleanContext.EvalBinGate(AND, zero, one), &result);
        if (result != 0)
            throw std::runtime_error("FHEW AND check failed");
        booleanContext.Decrypt(secret, booleanContext.EvalBinGate(OR, zero, one), &result);
        if (result != 1)
            throw std::runtime_error("FHEW OR check failed");
        std::cout << "FHEW AND/OR passed (STD128)." << std::endl;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
