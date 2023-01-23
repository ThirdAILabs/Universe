#pragma once
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <string>

namespace thirdai::licensing {

class FinegrainedAccessToken {
 public:
  // Creates a general FinegrainedAccessToken. This will throw an error
  // if the user does not have a full access entitlement. If licensing is
  // disabled, this will always succeed.
  FinegrainedAccessToken();

  // Creates a FinegrainedAccessToken corresponding to a passed in training
  // file path. If the user does not have an entitlement allowing them to
  // train on the file, this will throw an error. If licensing is
  // disabled, this will always succeed.
  explicit FinegrainedAccessToken(const std::string& train_file_path);

  void verifyCanSaveAndLoad() const {
    if (!_can_save_and_load) {
      throw exceptions::LicenseCheckException(
          "Cannot save or load with this license");
    }
  }

 private:
  bool _can_save_and_load;
};

// If license checking is enabled, verifies the license is valid and throws an
// exception otherwise. If licensing checking is disabled, this is a NOOP.
void checkLicense();

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