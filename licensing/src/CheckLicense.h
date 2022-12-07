#pragma once
#include <optional>
#include <string>

namespace thirdai::licensing {

void checkLicense();

void setLicensePath(const std::string& license_path);

void activate(const std::string& api_key);

void deactivate();

}  // namespace thirdai::licensing