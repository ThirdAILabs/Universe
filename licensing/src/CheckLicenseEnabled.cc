
#include "CheckLicense.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/Utils.h>
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/LicenseMethod.h>
#include <licensing/src/methods/file/FileMethod.h>
#include <licensing/src/methods/file/SignedLicense.h>
#include <licensing/src/methods/heartbeat/Heartbeat.h>
#include <licensing/src/methods/heartbeat/ServerMethod.h>
#include <licensing/src/methods/keygen/KeyMethod.h>
#include <licensing/src/methods/keygen/KeygenCommunication.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

// This file is only linked when the feature flag THIRDAI_CHECK_LICENSE is True

namespace thirdai::licensing {

// TODO(Kartik): Decide if we want to refactor these into a single
// std::unique_ptr<LicensingMethod> with a getEntitlements call. We would then
// set this in each call that starts licensing, and check the license there
// immediately upon the time it is set.

std::unique_ptr<LicenseMethod> _licensing_method = nullptr;

void checkLicense() {
#pragma message("THIRDAI_CHECK_LICENSE is defined, adding license checking code")  // NOLINT

  if (_licensing_method != nullptr) {
    _licensing_method->checkLicense();
  }

  throw exceptions::LicenseCheckException(
      "Please first call either licensing.set_path, "
      "licensing.start_heartbeat, or licensing.activate.");
}

Entitlements entitlements() { return _licensing_method->getEntitlements(); }

void activate(std::string api_key) {
  keygen::KeyMethod _licensing_method(std::move(api_key));
}

void startHeartbeat(std::string heartbeat_url,
                    std::optional<uint32_t> heartbeat_timeout) {
  heartbeat::ServerMethod _licensing_method(std::move(heartbeat_url),
                                            heartbeat_timeout);
}

void setLicensePath(std::string license_path, bool verbose) {
  file::FileMethod _licensing_method(std::move(license_path), verbose);
}

void deactivate() { _licensing_method = nullptr; }

LicenseState getLicenseState() { return _licensing_method->getLicenseState(); }

void setLicenseState(const LicenseState& state) {
  if (state.key_state) {
    activate(state.key_state.value());
  }
  if (state.server_state) {
    auto server_state_value = state.server_state.value();
    startHeartbeat(server_state_value.first, server_state_value.second);
  }
  if (state.file_state) {
    setLicensePath(state.file_state.value());
  }
}

}  // namespace thirdai::licensing