#include "License.h"
#include "LicenseWithSignature.h"
#include <cryptopp/files.h>
#include <map>
#include <unordered_map>

using thirdai::licensing::License;
using thirdai::licensing::LicenseWithSignature;

int main(int32_t argc, const char** argv) {
  if (argc < 5 || argc % 2 == 0) {
    std::cerr
        << "Invalid args, usage: ./create_signed_license "
           "private_key_file public_key_file output_file num_days [key value]*"
        << std::endl;
    return 1;
  }

  uint32_t num_args = argc;

  std::string private_key_file(argv[1]);
  std::string public_key_file(argv[2]);
  std::string output_file(argv[3]);
  int64_t num_days = std::stoi(argv[4]);
  std::map<std::string, std::string> metadata;
  for (uint32_t i = 5; i < num_args; i+=2) {
    std::string key(argv[i]);
    std::string val(argv[i + 1]);
    metadata.at(key) = val;
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
  LicenseWithSignature license_with_signature(license, private_key);

  // Write license with signature to file
  license_with_signature.serializeToFile(output_file);

  // Read the license back
  LicenseWithSignature read_from_file =
      LicenseWithSignature::deserializeFromFile(output_file);

  // Make sure the public key works
  if (!read_from_file.verify(public_key)) {
    std::cout << "Was not able to verify license with the public key, do not "
                 "use this license file!"
              << std::endl;
    exit(1);
  } else {
    std::cout << "Was able to verify license with the public key!" << std::endl;
  }

  std::cout << "Saved license " << read_from_file.get_license().toString()
            << std::endl;

  return 0;
}
