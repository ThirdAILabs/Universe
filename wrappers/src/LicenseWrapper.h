#pragma once
#include <optional>
#include <string>

namespace thirdai::licensing {

class LicenseWrapper {
 public:
  static void checkLicense();

  static void setLicensePath(const std::string& license_path);

 private:
  // This is nullopt unless the user sets a path, in which case it will be the
  // path the user sets.
  static std::optional<std::string> _license_path;
};

}  // namespace thirdai::licensing