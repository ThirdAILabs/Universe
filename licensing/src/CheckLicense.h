#pragma once
#include <dataset/src/DataSource.h>
#include <exceptions/src/Exceptions.h>
#include <optional>
#include <string>

namespace thirdai::licensing {

/**
 * This class is required by methods in the Bolt and Flash APIs that we want
 * to restrict access to in a pattern similar to FastAPI dependency injection.
 * Since the constructors of FinegrainedAccessToken themselves throw an
 * exception if the user doesn't have the correct permissions, we force the user
 * to callers of methods that take in FinegrainedAccessToken to have
 * permissions.
 */
class FinegrainedAccessToken {
 public:
  /** Creates a general FinegrainedAccessToken. This will throw an error
   * if the user does not have a full access entitlement. If licensing is
   * disabled, this will always succeed.
   */
  FinegrainedAccessToken();

  /**
   * Creates a FinegrainedAccessToken corresponding to a passed in training
   * file path. If the user does not have an entitlement allowing them to
   * train on the file, this will throw an error. If licensing is
   * disabled, this will always succeed. To prevent unauthorized API use, you
   * should only use this access token to train models with the passed in
   * training file.
   */
  explicit FinegrainedAccessToken(const std::string& train_file_path);

  void verifyCanTrain() const {
    // For now this is a NOOP because we always allow training if the
    // FinegrainedAccessToken was created successfully
  }
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

// If the user has the full access entitlement or license checking is disabled,
// this is a NOOP. Otherwise, this throws an exception.
void verifyCanSaveAndLoad();

}  // namespace thirdai::licensing