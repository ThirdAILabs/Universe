#include "License.h"
#include "SignedLicense.h"
#include <cryptopp/files.h>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <unordered_map>

using thirdai::licensing::License;
using thirdai::licensing::SignedLicense;

int main(int32_t argc, const char** argv) {
  if (argc < 5 || argc % 2 == 0) {
    std::cerr
        << "Invalid args, usage: ./create_signed_license "
           "private_key_file public_key_file output_file num_days [key value]*"
        << std::endl;
    return 1;
  }
  try {
    uint32_t num_args = argc;

    std::string private_key_file(argv[1]);
    std::string public_key_file(argv[2]);
    std::string output_file(argv[3]);
    int64_t num_days = std::stoi(argv[4]);
    std::map<std::string, std::string> metadata;
    for (uint32_t i = 5; i < num_args; i += 2) {
      std::string key(argv[i]);
      std::string val(argv[i + 1]);
      metadata[key] = val;
    }

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

    // Create and sign license
    License license = License::createLicenseWithNDaysLeft(metadata, num_days);
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

    std::cout << "Saved license " << read_from_file.get_license().toString()
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Create license failed with: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
