#pragma once
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <string>

namespace thirdai::licensing {

/**
 * This token can be added as an argument of a method to ensure that the method
 * can only be invoked if a user has correct permissions. A valid token can only
 * be constructed if the user has a full access license or is using a dataset
 * that is allowed under their demo license. By requiring the token as an
 * argument to the method, we require the caller to construct a token, and thus
 * prevent the method from being called if one of these conditions is not met.
 */
class TrainPermissionsToken {
 public:
  /**
   * Creates a general TrainPermissionsToken. This will throw an error
   * if the user does not have a full access entitlement. If licensing is
   * disabled, this will always succeed.
   */
  TrainPermissionsToken();

  /**
   * Creates a TrainPermissionsToken corresponding to a passed in training
   * data source. If the user does not have an entitlement allowing them to
   * train on the data source, this will throw an error. If licensing is
   * disabled, this will always succeed. To prevent unauthorized API use, you
   * should only use this access token to train models with the passed in
   * training file.
   */
  explicit TrainPermissionsToken(const dataset::DataSourcePtr& source);
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