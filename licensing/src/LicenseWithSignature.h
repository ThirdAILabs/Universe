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
#include <sys/types.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#if defined __linux__ || defined __APPLE__
#include <pwd.h>
#endif
#include <unistd.h>
#include <utility>

namespace thirdai::licensing {

const std::string PUBLIC_KEY_BASE_64 =
    "MIIBojANBgkqhkiG9w0BAQEFAAOCAY8AMIIBigKCAYEAsIv9g8w+"
    "DLlepzpE02luu6lV2DY7g5N0cnqhbaoArE5UOEiKK2EFPCeQTp8+TkYk64/"
    "ieMab4CoIU3ZmVp5GUyKkWsLJhDUE3dXJrLhIDTg7HFr6qwrFDosRWI26grq+"
    "CFPsiVLTjlJCd+7sv1EtR5TPhympKAKRbUI1pffnK8QTJ8F5Bfg/"
    "1tLHk3lpUp4vF90se0TWgmXe7CW6GtWeXqiwsfzK9IzkgLbX4DQJnyIRPS9MLoQr/"
    "nSws7jMPDtUIuSjUIOQojxIhxTO5iL+"
    "mfiV2h7nRLMtJM6lLKmrDK09sE4geE8zJytCcP1l15s7gZy7g7i1mwrpfiulmfNVvDj0LoKYD2"
    "nx1mj+gCgnUasqLWILNUXgV19eGGLd23+"
    "hc7NzF10KFVXIcLebrG7o6WfFY5NSYu2pDzialgpCXmiysyIKj/HXY1hpbi0/dMII/"
    "lVN2QhDb5zTVIjzBr+kMuJ9dNNl9Sn4eso+dMNjQrQ2F9WvcgS1ZQ4Ju/5qOZrRAgMBAAE=";

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

  /**
   * Checks for a license file in
   * 1. env(THIRDAI_LICENSE_PATH)
   * 2. ~/license.serialized
   * 3. {cwd}/license.serialized
   * Uses the first file found.
   * If no license is found, we throw an error.
   * If a license is found we verify it with the passed in public key,
   * then check whether it has expired. If either check fails we throw an error.
   * Otherwise we just return.
   */
  static void findVerifyAndCheckLicense() {
    std::vector<std::string> license_file_name_options;

    auto optional_license_environment_path =
        get_license_path_from_environment();
    if (!optional_license_environment_path->empty()) {
      license_file_name_options.push_back(
          optional_license_environment_path.value());
    }

    auto optional_home_dir = get_home_directory();
    if (!optional_home_dir->empty()) {
      license_file_name_options.push_back(optional_home_dir.value() +
                                          "/license.serialized");
    }

    auto optional_current_dir = get_current_directory();
    if (!optional_current_dir->empty()) {
      license_file_name_options.push_back(optional_current_dir.value() +
                                          "/license.serialized");
    }

    std::optional<std::pair<LicenseWithSignature, std::string>>
        license_with_file;
    for (const std::string& license_file_name : license_file_name_options) {
      if (can_access_file(license_file_name)) {
        license_with_file = {deserializeFromFile(license_file_name),
                             license_file_name};
        break;
      }
    }
    if (!license_with_file.has_value()) {
      throw thirdai::exceptions::LicenseCheckException(
          "no license file found. Go to https://thirdai.com/try-bolt to get a "
          "license.");
    }

    CryptoPP::RSA::PublicKey public_key;
    CryptoPP::StringSource ss(PUBLIC_KEY_BASE_64, true,
                              new CryptoPP::Base64Decoder);
    public_key.BERDecode(ss);

    if (!license_with_file->first.verify(public_key)) {
      throw thirdai::exceptions::LicenseCheckException(
          "license verification failure using license file " +
          license_with_file->second +
          ". Go to https://thirdai.com/try-bolt to get a valid license.");
    }

    if (license_with_file->first.get_license().isExpired()) {
      throw thirdai::exceptions::LicenseCheckException(
          "license file " + license_with_file->second +
          "expired. Go to "
          "https://thirdai.com/try-bolt to renew your license.");
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

    return path.u8string();
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