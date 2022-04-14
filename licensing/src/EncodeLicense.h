#pragma once

#include "License.h"
#include <string>
#include <cryptopp/rsa.h>

namespace thirdai::licensing {

inline std::string encodeLicense(const License& license, const CryptoPP::RSA::PrivateKey& private_key) {
  (void)license;
  (void)private_key;
  return "";
}

}  // namespace thirdai::licensing