#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/entitlements/RestrictionTree.h>
#include <optional>
#include <stdexcept>
#include <unordered_set>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is False

namespace thirdai::licensing {

void checkLicense() {}

Entitlements entitlements() { return Entitlements({FULL_ACCESS_ENTITLEMENT}); }

// The following functions have NOLINT because their arguments are pass-by-value
// even though they are not used. The values are used in CheckLicenseEnabled.cc,
// which requires the method signature to be pass by value for these functions
// as well
void activate(std::string api_key) { (void)api_key; }  // NOLINT

void startHeartbeat(std::string heartbeat_url,  // NOLINT
                    const std::optional<uint32_t>& heartbeat_timeout) {
  (void)heartbeat_url;
  (void)heartbeat_timeout;
}

void setLicensePath(std::string license_path, bool verbose) {  // NOLINT
  (void)license_path;
  (void)verbose;
}

void deactivate() {}

LicenseState getLicenseState() { return {}; }

void setLicenseState(const LicenseState& state) { (void)state; }

}  // namespace thirdai::licensing