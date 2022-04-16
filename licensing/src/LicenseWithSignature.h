#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include "License.h"
#include <cryptopp/files.h>
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <exceptions/src/Exceptions.h>
#include <exception>
#include <fstream>
#include <optional>
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

  void writeToFile(const std::string& path) {
    std::ofstream filestream(path, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<LicenseWithSignature> readFromFile(
      const std::string& path) {
    std::ifstream filestream(path, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<LicenseWithSignature> serialize_into(
        new LicenseWithSignature());
    iarchive(*serialize_into);
    return serialize_into;
  }

  /** Checks for a license file named license.serialized in...
   *  1. /home/thirdai/
   *  2. The current directory
   * Using a public key named license-public-key.der in...
   *  1. /home/thirdai
   *  2. The current directory
   * If no license is found, throws an error. If no public key is found, throws
   * an error. We check 1. and use it if we find something, and otherwise
   * check 2. If a license is found we verify it with the passed in public key,
   * then check whether it has expired. If either check fails we throw an error.
   * Otherwise we just return.
   */
  static void findVerifyAndCheckLicense() {
    std::vector<std::string> license_file_name_options = {
        "/home/thirdai/license.serialized", "license.serialized"};
    std::vector<std::string> public_key_file_name_options = {
        "/home/thirdai/license-public-key.der", "license-public-key.der"};

    std::unique_ptr<LicenseWithSignature> license;
    for (const std::string& license_file_name : license_file_name_options) {
      if (can_access_file(license_file_name)) {
        license = readFromFile(license_file_name);
        break;
      }
    }
    if (!license) {
      throw thirdai::exceptions::LicenseCheckException("no license file found");
    }

    std::optional<CryptoPP::RSA::PublicKey> public_key;
    for (const std::string& public_key_file_name :
         public_key_file_name_options) {
      if (can_access_file(public_key_file_name)) {
        CryptoPP::RSA::PublicKey load_into;
        {
          CryptoPP::FileSource input(public_key_file_name.c_str(), true);
          load_into.BERDecode(input);
        }
        public_key = load_into;
        break;
      }
    }
    if (!public_key) {
      throw thirdai::exceptions::LicenseCheckException(
          "no public key file found");
    }

    if (!license->verify(public_key.value())) {
      throw thirdai::exceptions::LicenseCheckException(
          "license verification failure");
    }

    if (license->get_license().isExpired()) {
      throw thirdai::exceptions::LicenseCheckException("license expired");
    }
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

  static bool can_access_file(const std::string& fileName) {
    std::ifstream infile(fileName);
    return infile.good();
  }

  License _license;
  std::string _signature;
};

}  // namespace thirdai::licensing