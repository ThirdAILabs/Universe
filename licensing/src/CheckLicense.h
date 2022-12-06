#pragma once
#include <optional>
#include <string>

namespace thirdai::licensing {

class CheckLicense {
 public:
  static void checkLicenseWrapper();

  static void setLicensePath(const std::string& license_path);

  static void activate(const std::string& api_key);

  static void deactivate();
};

}  // namespace thirdai::licensing