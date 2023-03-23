#include "License.h"
#include "SignedLicense.h"
#include <cryptopp/files.h>
#include <licensing/src/entitlements/EntitlementTree.h>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <string>

namespace thirdai::licensing {

std::map<std::string, std::string> getEntitlementsFromUser() {
  std::map<std::string, std::string> entitlements;
  std::string input;

  std::cout << "We will now begin the license generation process by asking a "
               "series of questions. "
            << std::endl;
  std::cout << "First, do you want a full access license? Please answer y or n "
               "(anything else will be treated as no)."
            << std::endl;
  std::cin >> input;

  if (input == "y") {
    entitlements[FULL_ACCESS_ENTITLEMENT] = FULL_ACCESS_ENTITLEMENT;
    return entitlements;
  }

  // We don't mess around with per-dataset licenses with file licenses
  // (at least for now)
  entitlements[FULL_DATASET_ENTITLEMENT] = FULL_DATASET_ENTITLEMENT;

  std::cout << "Should the user be able to save and load models?" << std::endl;
  std::cin >> input;

  if (input == "y") {
    entitlements[LOAD_SAVE_ENTITLEMENT] = LOAD_SAVE_ENTITLEMENT;
  }

  std::cout << "What should be the maximum output dimension for models? "
               "Please choose a number < 2^64. Input m to choose 2^64 - 1."
            << std::endl;
  std::cin >> input;

  uint64_t max_output_dim =
      input == "m" ? std::numeric_limits<uint64_t>::max() : std::stoul(input);

  entitlements[MAX_OUTPUT_DIM_ENTITLEMENT_START] =
      MAX_OUTPUT_DIM_ENTITLEMENT_START + " " + std::to_string(max_output_dim);

  std::cout << "What should be the maximum number of samples (across epochs) "
               "a model should be able to be trained on? Please choose a "
               "number < 2^64. Input m to choose 2^64 - 1."
            << std::endl;
  std::cin >> input;

  uint64_t max_num_samples =
      input == "m" ? std::numeric_limits<uint64_t>::max() : std::stoul(input);

  entitlements[MAX_TRAIN_SAMPLES_ENTITLEMENT_START] =
      MAX_TRAIN_SAMPLES_ENTITLEMENT_START + " " +
      std::to_string(max_num_samples);

  return entitlements;
}

}  // namespace thirdai::licensing

using thirdai::licensing::License;
using thirdai::licensing::SignedLicense;

int main(int32_t argc, const char** argv) {
  if (argc < 5 || argc % 2 == 0) {
    std::cerr << "Invalid args, usage: ./create_signed_license "
                 "private_key_file public_key_file output_file num_days "
              << std::endl;
    return 1;
  }
  try {
    uint32_t num_args = argc;

    std::string private_key_file(argv[1]);
    std::string public_key_file(argv[2]);
    std::string output_file(argv[3]);
    int64_t num_days = std::stoi(argv[4]);

    // Read keys from file
    CryptoPP::RSA::PrivateKey private_key;
    {
      CryptoPP::FileSource input(private_key_file.c_str(), true);
      private_key.BERDecode(input);
    }
    CryptoPP::RSA::PublicKey public_key;
    {
      CryptoPP::FileSource input(public_key_file.c_str(), true);
      public_key.BERDecode(input);
    }

    auto entitlements = thirdai::licensing::getEntitlementsFromUser();

    // Create and sign license
    License license =
        License::createLicenseWithNDaysLeft(entitlements, num_days);
    SignedLicense license_with_signature(license, private_key);

    // Write license with signature to file
    try {
      license_with_signature.serializeToFile(output_file);
    } catch (const std::exception& e) {
      std::cerr << "Failed to write license to file: " << e.what() << std::endl;
      return 1;
    }

    // Read the license back
    SignedLicense read_from_file;
    try {
      read_from_file = SignedLicense::deserializeFromFile(output_file);
    } catch (const std::exception& e) {
      std::cerr << "Failed to read license from file: " << e.what()
                << std::endl;
      return 1;
    }

    // Make sure the public key works
    if (!read_from_file.verify(public_key)) {
      std::cout
          << "Was not able to verify license with the public key, deleting "
             " the created file."
          << std::endl;
      std::filesystem::remove(output_file);
      return 1;
    }
    std::cout << "Was able to verify license with the public key!" << std::endl;

    std::cout << "Saved license " << read_from_file.getLicense().toString()
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Create license failed with: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
