#pragma once
#include <licensing/src/entitlements/Entitlements.h>
#include <licensing/src/methods/LicenseMethod.h>
#include <licensing/src/methods/heartbeat/Heartbeat.h>

namespace thirdai::licensing::heartbeat {

class LocalServerMethod final : public LicenseMethod {
 public:
  LocalServerMethod(std::string heartbeat_url,
                    std::optional<uint32_t> heartbeat_timeout);

  void checkLicense() final;

  LicenseState getLicenseState() final;

 private:
  std::string _heartbeat_url;
  std::optional<uint32_t> _heartbeat_timeout;
  HeartbeatThread _heartbeat_thread;
};

}  // namespace thirdai::licensing::heartbeat