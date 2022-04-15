#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include "License.h"
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <exception>
#include <fstream>
#include <utility>

namespace thirdai::licensing {

class LicenseWithSignature {
 public:
  // This is public because it is a top level serialization target, only
  // call this if you are creating an object to serialize into
  LicenseWithSignature() : _license(std::map<std::string, std::string>(), 0){};

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

  // Returns true if the content of the license can be verified with the public
  // key and false otherwise
  bool verify(const CryptoPP::RSA::PublicKey& public_key) {
    // See https://cryptopp.com/wiki/RSA_Cryptography for more details

    std::string license_state_to_sign = _license.toString();

    CryptoPP::RSASSA_PKCS1v15_SHA_Verifier verifier(public_key);

    try {
      CryptoPP::StringSource ss2(
          license_state_to_sign + _signature, true,
          new CryptoPP::SignatureVerificationFilter(
              verifier, NULL,
              CryptoPP::SignatureVerificationFilter::
                  THROW_EXCEPTION)  // SignatureVerificationFilter
      );                            // StringSource
    } catch (const CryptoPP::SignatureVerificationFilter::
                 SignatureVerificationFailed& e) {
      return false;
    }

    return true;
  }

  void writeLicenseAndSignatureToFile(const std::string& path) {
    std::ofstream filestream(path, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<LicenseWithSignature> readLicenseAndSignatureFromFile(
      const std::string& path) {
    std::ifstream filestream(path, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<LicenseWithSignature> serialize_into(
        new LicenseWithSignature());
    iarchive(*serialize_into);
    return serialize_into;
  }

  const License& get_license() { return _license; }

  // For now this is just used for testing
  void set_license(License new_license) { _license = std::move(new_license); }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_signature, _license);
  }

  License _license;
  std::string _signature;
};

}  // namespace thirdai::licensing