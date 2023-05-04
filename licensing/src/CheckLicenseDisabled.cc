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

void activate(std::string api_key) { (void)api_key; } // NOLINT

void deactivate() {}

void startHeartbeat(std::string heartbeat_url, // NOLINT
                    const std::optional<uint32_t>& heartbeat_timeout) {
  (void)heartbeat_url;
  (void)heartbeat_timeout;
}

void endHeartbeat() {}

// TODO(Kartik): Add explanation for why we have this nolint
void setLicensePath(std::string license_path, bool verbose) { // NOLINT
  (void)license_path;
  (void)verbose;
}

LicenseState getLicenseState() { return {}; }

}  // namespace thirdai::licensing