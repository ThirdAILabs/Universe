#pragma once
#include <optional>
#include <string>

namespace thirdai::licensing {

class LicenseWrapper {
 public:
  static void checkLicense(const std::optional<std::string>& license_path);
};

}  // namespace thirdai::licensing