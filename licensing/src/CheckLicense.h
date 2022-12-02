#pragma once

#include <licensing/src/file/SignedLicense.h>
#include <licensing/src/keygen/KeygenCommunication.h>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::licensing {

inline void checkLicense(const std::optional<std::string>& access_key,
                         const std::optional<std::string>& license_path) {
  if (access_key.has_value()) {
    KeygenCommunication::verifyWithKeygen(*access_key);
    return;
  }

  SignedLicense::findVerifyAndCheckLicense(license_path);
}

}  // namespace thirdai::licensing