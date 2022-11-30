#pragma once
#include <optional>
#include <string>

namespace thirdai::licensing {

class LicenseWrapper {
 public:
  static void checkLicense();

  static void setLicensePath(const std::string& license_path);

  static void activate(const std::string& api_key);

 private:
  // This is nullopt unless the user sets a path, in which case it will be the
  // path the user sets.
  static std::optional<std::string> _license_path;

  // This is nullopt unless the user activates with an api key, in which case it
  // will be the api key the user passes in.
  static std::optional<std::string> _api_key;
};

}  // namespace thirdai::licensing