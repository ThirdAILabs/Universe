#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include "License.h"
#include <cryptopp/base64.h>
#include <cryptopp/files.h>
#include <cryptopp/filters.h>
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <sys/types.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <unordered_set>
#if defined __linux__ || defined __APPLE__
#include <pwd.h>
#include <unistd.h>
#endif
#include <utility>

namespace thirdai::licensing {

class SignedLicense {
 public:
  /**
   * Verifies the passed in license file with the stored public key,
   * then checks whether it has expired. If either fails we throw an error.
   * Otherwise we return the entitlements found in the license file.
   */
  static Entitlements verifyPathAndGetEntitlements(
      const std::string& license_path, bool verbose) {
    SignedLicense license = getLicenseFromFile(license_path);

    verifyAndCheckLicense(license, license_path);

    if (verbose) {
      std::cout << license.getLicense().toHumanString() << std::endl;
    }

    return license.getLicense().entitlements();
  }

  // This is public because it is a top level serialization target, only
  // call this if you are creating an object to serialize into
  SignedLicense() : _license(std::map<std::string, std::string>(), 0){};

  SignedLicense(License license, const CryptoPP::RSA::PrivateKey& private_key)
      : _license(std::move(license)) {
    // See https://cryptopp.com/wiki/RSA_Cryptography for more details

    std::string license_state_to_sign = _license.toVerifiableString();

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
  bool verify(const CryptoPP::RSA::PublicKey& public_key) const {
    // See https://cryptopp.com/wiki/RSA_Cryptography for more details

    std::string license_state_to_sign = _license.toVerifiableString();

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

  static SignedLicense deserializeFromFile(const std::string& path) {
    std::ifstream filestream(path, std::ios::binary);
    SignedLicense serialize_into;
    {
      cereal::BinaryInputArchive iarchive(filestream);
      iarchive(serialize_into);
    }
    return serialize_into;
  }

  const License& getLicense() const { return _license; }

  // For now this is just used for testing
  void setLicense(License new_license) { _license = std::move(new_license); }

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_signature, _license);
  }

  static inline const std::string PUBLIC_KEY_BASE_64 =
      "MIIBojANBgkqhkiG9w0BAQEFAAOCAY8AMIIBigKCAYEAw4ZXDhvzjwpN6N2HaX64H7KMAZGg"
      "nyEvIvWYHNgUEl5E4C1DsfzeDCZNU1xvAzwssiUUVN3RQJ1XPESIMZH9eO9TCTmVhGAo407m"
      "phJ8vDm7uQw6i6mpvxvYDY0HuUyhGGWAzN1wooBwH82IUfIjrhc2S2VEpSLBS7wHqO2doRiE"
      "09ormgqPRFHh63rWw/83DGWXKxeKiQG0Oq2dBY90ZkPO1npAjVJAM7KUqv/"
      "kMEpz9CXBEaNTewKW0pG7WypyGp5UmeGDjoyivD7BaVopRSNh12H2FLvKDyahiJlKRW99R4e"
      "5muqc31DLlYeVULJIZDC3zpv/"
      "TXn5IOnZ0ftw9H8skLOp+"
      "jHvUvf5UGITjlZaXbeGxRvtdyMayCDar1DnkwKmquzYPT3SOjIyAV9C9kp/QGCndgQzc8/"
      "bPlFPUhv7J99gfFFzjPefpfRkB9z/"
      "x0AMN2a0j7V6qlTUDLdRWapRX92CTJU0cUuKdWXh4+TE+"
      "narN9tYVp5MpTfgfGorAgMBAAE=";

  static SignedLicense getLicenseFromFile(const std::string& license_file) {
    if (!can_access_file(license_file)) {
      throw exceptions::LicenseCheckException(
          "Cannot access the passed in license file " + license_file);
    }
    return deserializeFromFile(license_file);
  }

  // Verifies the license with the stored public key. If the verification fails,
  // throws an error. If the verication is succesfull but the license has
  // expired, throws an error.
  static void verifyAndCheckLicense(const SignedLicense& license,
                                    const std::string& license_file) {
    CryptoPP::RSA::PublicKey public_key;
    CryptoPP::StringSource ss(PUBLIC_KEY_BASE_64, true,
                              new CryptoPP::Base64Decoder);
    public_key.BERDecode(ss);

    if (!license.verify(public_key)) {
      throw thirdai::exceptions::LicenseCheckException(
          "license verification failure using license file " + license_file +
          ". Go to https://thirdai.com/try-bolt to get a valid license.");
    }

    if (license.getLicense().isExpired()) {
      throw thirdai::exceptions::LicenseCheckException(
          "the following license file is expired: " + license_file +
          ". Go to https://thirdai.com/try-bolt to renew your license.");
    }
  }

  static bool can_access_file(const std::string& fileName) {
    std::ifstream infile(fileName);
    return infile.good();
  }

  static std::optional<std::string> get_license_path_from_environment() {
    const char* license_env_path = std::getenv("THIRDAI_LICENSE_PATH");
    if (license_env_path == NULL) {
      return {};
    }
    // This copies the char* into a string object, so no we are not deleting
    // the memory of the environment variable (which is good!)
    return license_env_path;
  }

  static std::optional<std::string> get_current_directory() {
    auto path = std::filesystem::current_path();
    if (path.empty()) {
      return {};
    }

    return path.string();
  }

  static std::optional<std::string> get_home_directory() {
#if defined __linux__ || defined __APPLE__
    struct passwd* pw = getpwuid(getuid());
    if (pw == NULL) {
      return std::nullopt;
    }
    char* dir = getpwuid(getuid())->pw_dir;
    if (dir == NULL) {
      return std::nullopt;
    }
    return std::string(dir);
#elif _WIN32
    return get_home_directory_windows();
#endif
  }

  static std::optional<std::string> get_home_directory_windows() {
    char* dir = std::getenv("USERPROFILE");
    if (dir == NULL) {
      char* home_drive = std::getenv("HOMEDRIVE");
      char* home_path = std::getenv("HOMEPATH");
      if (home_drive == NULL || home_path == NULL) {
        return std::nullopt;
      }
      // This copies the char* into a string object, so no we are not deleting
      // the memory of the environment variable (which is good!)
      return std::string(home_drive) + std::string(home_path);
    }
    return dir;
  }

  License _license;
  std::string _signature;
};

}  // namespace thirdai::licensing