#pragma once
#include <dataset/src/DataSource.h>
#include <optional>
#include <string>

namespace thirdai::licensing {

// If license checking is enabled, verifies the license is valid and throws an
// exception otherwise. If licensing checking is disabled, this is a NOOP.
void checkLicense();

// If license checking is enabled, verifies that the file is authorized under
// the license. If license checking is disabled, this is a NOOP.
void verifyAllowedDataset(const std::optional<std::string>& filename);

// License verification method 1: Keygen api key
void activate(const std::string& api_key);
void deactivate();

// License verification method 2: heartbeat
void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout);
void endHeartbeat();

// License verification method 3: license file
void setLicensePath(const std::string& license_path);

}  // namespace thirdai::licensing