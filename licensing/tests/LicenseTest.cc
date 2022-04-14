#include <gtest/gtest.h>
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <licensing/src/DecodeLicense.h>
#include <licensing/src/EncodeLicense.h>
#include <licensing/src/License.h>
#include <unordered_map>

namespace thirdai::licensing {

TEST(LicenseTest, EncodeDecodeTest) {
  // See https://www.cryptopp.com/wiki/RSA_Cryptography (much of this code is
  // taken from their examples)

  // Pseudo Random Number Generator
  CryptoPP::AutoSeededRandomPool rng;

  // Generate Parameters
  CryptoPP::InvertibleRSAFunction params;
  params.GenerateRandomWithKeySize(rng, 3072);

  // Create Keys
  CryptoPP::RSA::PrivateKey private_key(params);
  CryptoPP::RSA::PublicKey public_key(params);  // NOLINT

  // Create License
  std::unordered_map<std::string, std::string> metadata{
      {"company", "abc"}, {"person", "def"}, {"machine type", "ghi"}};
  License license = License::createLicenseWithNDaysLeft(metadata, 30);

  // Encode License
  auto encoded_license = encodeLicense(license, private_key);

  // Decode License
  License decoded_license = decodeLicense(encoded_license, public_key);

  // Assert License was decode correctly
  ASSERT_EQ(decoded_license.getExpireTimeMillis(), license.getExpireTimeMillis());
  ASSERT_EQ(decoded_license.getMetadataValue("company"), "abc");
  ASSERT_EQ(decoded_license.getMetadataValue("person"), "def");
  ASSERT_EQ(decoded_license.getMetadataValue("machine_type"), "ghi");
}

}  // namespace thirdai::licensing