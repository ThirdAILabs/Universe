#include <gtest/gtest.h>
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <licensing/src/License.h>
#include <licensing/src/LicenseWithSignature.h>
#include <unordered_map>

namespace thirdai::licensing {

TEST(LicenseTest, SignAndVerifyTest) {
  // See https://www.cryptopp.com/wiki/RSA_Cryptography (much of this code is
  // taken from their examples)

  // Create keys
  CryptoPP::AutoSeededRandomPool rng;
  CryptoPP::InvertibleRSAFunction params;
  params.GenerateRandomWithKeySize(rng, 3072);

  CryptoPP::RSA::PrivateKey private_key(params);
  CryptoPP::RSA::PublicKey public_key(params);  // NOLINT

  // Create license
  std::unordered_map<std::string, std::string> metadata{
      {"company", "abc"}, {"person", "def"}, {"machine type", "ghi"}};
  License license = License::createLicenseWithNDaysLeft(metadata, 30);

  // Create signed license
  LicenseWithSignature signed_license(license, private_key);

  // Verify License
  signed_license.verify(public_key);
}

}  // namespace thirdai::licensing