#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <optional>
#include <stdexcept>
#include <unordered_set>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is False

namespace thirdai::licensing {

TrainPermissionsToken::TrainPermissionsToken(
    const std::string& train_file_path) {
  (void)train_file_path;
}

TrainPermissionsToken::TrainPermissionsToken() {}

void checkLicense() {}

void verifyAllowedDataset(const std::optional<std::string>& filename) {
  (void)filename;
}

void activate(const std::string& api_key) { (void)api_key; }

void deactivate() {}

void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout) {
  (void)heartbeat_url;
  (void)heartbeat_timeout;
}

void endHeartbeat() {}

void setLicensePath(const std::string& license_path) { (void)license_path; }

void disableForDemoLicenses() {}

}  // namespace thirdai::licensing