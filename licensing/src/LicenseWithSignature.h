#pragma once

#include "License.h"
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <utility>

namespace thirdai::licensing {

class LicenseWithSignature {
 public:
  // This is public because it is a top level serialization target, only
  // call this if you are creating an object to serialize into
  LicenseWithSignature();

  LicenseWithSignature(License license,
                       const CryptoPP::RSA::PrivateKey& private_key)
      : _license(std::move(license)) {
    // See https://cryptopp.com/wiki/RSA_Cryptography for more details

    std::string license_state_to_sign = _license.toString();

    // This is automatically seeded with the OS's randomness source
    CryptoPP::AutoSeededRandomPool rng;

    // These new objects are automatically deleted, it's the recommendec way to
    // use this library. Just defining the pipeline source -> filter -> sink
    // causes it to run.
    CryptoPP::RSASSA_PKCS1v15_SHA_Signer signer(private_key);
    CryptoPP::StringSource ss1(
        license_state_to_sign, true,
        new CryptoPP::SignerFilter(
            rng, signer,
            new CryptoPP::StringSink(_signature))  // SignerFilter
    );                                             // StringSource
  }

  // This will throw a HashVerificationFailed if the verification fails
  void verify(const CryptoPP::RSA::PublicKey& public_key) {
    // See https://cryptopp.com/wiki/RSA_Cryptography for more details

    std::string license_state_to_sign = _license.toString();

    CryptoPP::RSASSA_PKCS1v15_SHA_Verifier verifier(public_key);

    CryptoPP::StringSource ss2(
        license_state_to_sign + _signature, true,
        new CryptoPP::SignatureVerificationFilter(
            verifier, NULL,
            CryptoPP::SignatureVerificationFilter::
                THROW_EXCEPTION)  // SignatureVerificationFilter
    );                            // StringSource
  }

 private:
  License _license;
  std::string _signature;
};
}  // namespace thirdai::licensing