#pragma once
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <optional>
#include <string>
#include <licensing/src/methods/LicenseMethod.h>

namespace thirdai::licensing {

// ----- Methods used internally to check license permissions:

// If license checking is enabled, verifies the license is valid and throws an
// exception otherwise. If licensing checking is disabled, this is a NOOP.
// This also updates the current entitlements object. This should be called in
// all model constructors.
void checkLicense();

// Get the entitlements object for the current license
Entitlements entitlements();

// ------ Methods to activate and deactivate licenses:

// License verification method 1: Keygen api key
void activate(std::string api_key);
void deactivate();

// License verification method 2: heartbeat
void startHeartbeat(std::string heartbeat_url,
                    std::optional<uint32_t> heartbeat_timeout);

// License verification method 3: license file
void setLicensePath(const std::string& license_path, bool verbose = false);

LicenseState getLicenseState();

void setLicenseState(const LicenseState& state);

}  // namespace thirdai::licensing