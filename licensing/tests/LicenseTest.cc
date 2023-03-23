#include <gtest/gtest.h>
#include <cryptopp/osrng.h>
#include <cryptopp/rsa.h>
#include <licensing/src/methods/file/License.h>
#include <licensing/src/methods/file/SignedLicense.h>
#include <fstream>
#include <map>

namespace thirdai::licensing {

class LicenseTest : public testing::Test {
 public:
  static std::pair<CryptoPP::RSA::PrivateKey, CryptoPP::RSA::PublicKey>
  generateKeyPair() {
    // See https://www.cryptopp.com/wiki/RSA_Cryptography
    CryptoPP::AutoSeededRandomPool rng;
    CryptoPP::InvertibleRSAFunction params;
    params.GenerateRandomWithKeySize(rng, 3072);

    CryptoPP::RSA::PrivateKey private_key(params);
    CryptoPP::RSA::PublicKey public_key(params);  // NOLINT

    return {private_key, public_key};
  }

  static License createLicense(int64_t num_days = 30) {
    std::map<std::string, std::string> entitlements{
        {"Entitlement 1", "Entitlement 1"}, {"Entitlement 2", "Entitlement 2"}};
    return License::createLicenseWithNDaysLeft(entitlements, num_days);
  }
};

TEST(LicenseTest, SignVerifyTest) {
  // Create keys
  auto [private_key, public_key] = LicenseTest::generateKeyPair();

  // Create license
  auto license = LicenseTest::createLicense();

  // Create signed license
  SignedLicense signed_license(license, private_key);

  // Verify License
  signed_license.verify(public_key);
}

TEST(LicenseTest, SignSerializeDeserializeVerifyTest) {
  auto [private_key, public_key] = LicenseTest::generateKeyPair();

  // Create license
  auto license = LicenseTest::createLicense();

  // Create signed license
  SignedLicense signed_license(license, private_key);

  // Write to disk and then read from disk
  signed_license.serializeToFile("license.serialized");
  SignedLicense license_from_disk =
      SignedLicense::deserializeFromFile("license.serialized");

  // Verify License
  ASSERT_TRUE(license_from_disk.verify(public_key));
}

TEST(LicenseTest, SignModifyVerifyTest) {
  auto [private_key, public_key] = LicenseTest::generateKeyPair();

  // Create license
  auto license = LicenseTest::createLicense();

  // Create signed license
  SignedLicense signed_license(license, private_key);

  // Verifying with a different public key should fail
  auto [private_key_2, public_key_2] = LicenseTest::generateKeyPair();
  ASSERT_FALSE(signed_license.verify(public_key_2));

  // Modifying the license and then verifying with the original key should fail
  signed_license.setLicense(LicenseTest::createLicense(1000));
  ASSERT_FALSE(signed_license.verify(public_key));
}

}  // namespace thirdai::licensing