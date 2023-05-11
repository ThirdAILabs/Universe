#include "LocalServerMethod.h"

namespace thirdai::licensing::heartbeat {

LocalServerMethod::LocalServerMethod(std::string heartbeat_url,
                                     std::optional<uint32_t> heartbeat_timeout)
    : LicenseMethod(Entitlements({FULL_ACCESS_ENTITLEMENT}),
                    licensing::LicenseMethodType::SERVER),
      _heartbeat_url(std::move(heartbeat_url)),
      _heartbeat_timeout(heartbeat_timeout),
      _heartbeat_thread(HeartbeatThread(_heartbeat_url, _heartbeat_timeout)) {
  _heartbeat_thread.verify();
};

void LocalServerMethod::checkLicense() { _heartbeat_thread.verify(); }

LicenseState LocalServerMethod::getLicenseState() {
  LicenseState license_state;
  license_state.local_server_state = {_heartbeat_url, _heartbeat_timeout};
  return license_state;
}

}  // namespace thirdai::licensing::heartbeat