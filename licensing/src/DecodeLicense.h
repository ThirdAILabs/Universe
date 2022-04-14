#pragma once

#include "License.h"
#include <string>
#include <cryptopp/rsa.h>

namespace thirdai::licensing {

inline License decodeLicense(const std::string& encoded_license,
                             const CryptoPP::RSA::PublicKey& public_key) {
  (void)encoded_license;
  (void)public_key;
  return License();
}

}  // namespace thirdai::licensing