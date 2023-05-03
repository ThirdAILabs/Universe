#pragma once
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/entitlements/Entitlements.h>
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
void activate(const std::string& api_key);
void deactivate();

// License verification method 2: heartbeat
void startHeartbeat(const std::string& heartbeat_url,
                    const std::optional<uint32_t>& heartbeat_timeout);
void endHeartbeat();

// License verification method 3: license file
void setLicensePath(const std::string& license_path, bool verbose = false);

struct LicenseState {
  std::optional<std::string> api_key_state;
  std::optional<std::pair<std::string, std::optional<uint32_t>>>
      heartbeat_state;
  std::optional<std::string> license_path_state;
};

LicenseState getLicenseState();

inline void setLicenseState(const LicenseState& state) {
  if (state.api_key_state) {
    activate(state.api_key_state.value());
  }
  if (state.heartbeat_state) {
    auto heartbeat_state_value = state.heartbeat_state.value();
    startHeartbeat(heartbeat_state_value.first, heartbeat_state_value.second);
  }
  if (state.license_path_state) {
    setLicensePath(state.license_path_state.value());
  }
}

}  // namespace thirdai::licensing