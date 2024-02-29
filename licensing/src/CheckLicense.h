#pragma once
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/LicenseMethod.h>
#include <optional>
#include <string>

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
#if !_WIN32
__attribute__((visibility("default")))
#endif
 void activate(std::string api_key);

// License verification method 2: heartbeat
void startHeartbeat(std::string heartbeat_url,
                    std::optional<uint32_t> heartbeat_timeout);

// License verification method 3: license file
void setLicensePath(std::string license_path, bool verbose = false);

// Deactivate any license method
void deactivate();

LicenseState getLicenseState();

void setLicenseState(const LicenseState& state);

}  // namespace thirdai::licensing