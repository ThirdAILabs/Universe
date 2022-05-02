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

    // These lines sign the License state for later verification.
    // The objects created with the "new" keyword are automatically deleted,
    // it's the recommended way use this library.
    // We don't need to do anything besides defining the
    // "source -> signer -> sink" pipeline using a nested builder pattern.
    // This is because the pumpAll boolean is set to true, so the source will
    // automatically be pumped into the SignerVerifier, which will in turn pump
    // the signature into the sink and therefore the _signature field. Thus
    // after these next lines _signature will contain the signature of the
    // license state and this object will be ready for serialization. See
    // https://cryptopp.com/wiki/RSA_Cryptography for more details.
    CryptoPP::RSASSA_PKCS1v15_SHA_Signer signer(private_key);
    CryptoPP::StringSource source(
        /* string = */ license_state_to_sign, /* pumpAll = */ true,
        new CryptoPP::SignerFilter(
            /* rng = */ rng, /* signer = */ signer,
            new CryptoPP::StringSink(
                /* output = */ _signature))  // SignerFilter
    );                                       // StringSource
  }

  // Returns true if the content of the license can be verified with the public
  // key and false otherwise
  bool verify(const CryptoPP::RSA::PublicKey& public_key) {
    // See https://cryptopp.com/wiki/RSA_Cryptography for more details

    std::string license_state_to_sign = _license.toString();

    // These lines try to verify the License with the signature field
    // and the passed in public key. Similar to in the constructor above, the
    // objects created with the "new" keyword are automatically deleted and the
    // "source -> verifier" pipeline automatically runs on construction.
    // The verificaiton process works as follows: the license state concatenated
    //  with the signature is verified by the VerificaitonFilter, which does the
    // verification with the public key it was constructed with.
    // If the verification fails the cryptopp library
    // throws an error, which we catch, and then we return false. See
    // https://cryptopp.com/wiki/RSA_Cryptography for more details.
    CryptoPP::RSASSA_PKCS1v15_SHA_Verifier verifier(public_key);
    try {
      CryptoPP::StringSource source(
          /* string = */ license_state_to_sign + _signature,
          /* pumpAll = */ true,
          new CryptoPP::SignatureVerificationFilter(
              /* verifier = */ verifier, /* attachment = */ NULL,
              /* flags = */
              CryptoPP::SignatureVerificationFilter::
                  THROW_EXCEPTION)  // SignatureVerificationFilter
      );                            // StringSource
    } catch (const std::exception& e) {
      return false;
    }

    return true;
  }

  void serializeToFile(const std::string& path) {
    std::ofstream filestream(path, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static LicenseWithSignature deserializeFromFile(const std::string& path) {
    std::ifstream filestream(path, std::ios::binary);
    LicenseWithSignature serialize_into;
    {
      cereal::BinaryInputArchive iarchive(filestream);
      iarchive(serialize_into);
    }
    return serialize_into;
  }

  /** Checks for a license file named license.serialized in...
   *  1. /home/thirdai/work (user added)
   *  2. /licenses (added automatically at Docker build time)
   * Using a public key named license-public-key.der in...
   *  1. /keys (added automatically at Docker build time)
   * If no license is found, throws an error. If no public key is found, throws
   * an error. If a license is found we verify it with the passed in public key,
   * then check whether it has expired. If either check fails we throw an error.
   * Otherwise we just return.
   */
  static void findVerifyAndCheckLicense() {
    std::vector<std::string> license_file_name_options = {
        "/home/thirdai/work/license.serialized",
        "/licenses/license.serialized"};
    std::vector<std::string> public_key_file_name_options = {
        "/keys/license-public-key.der"};

    std::optional<LicenseWithSignature> license;
    for (const std::string& license_file_name : license_file_name_options) {
      if (can_access_file(license_file_name)) {
        license = deserializeFromFile(license_file_name);
        break;
      }
    }
    if (!license.has_value()) {
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