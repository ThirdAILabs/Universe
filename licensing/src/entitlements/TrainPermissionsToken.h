#pragma once

#include <dataset/src/DataSource.h>
#include <licensing/src/CheckLicense.h>

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
  TrainPermissionsToken() { entitlements().verifyFullAccess(); }

  /**
   * Creates a TrainPermissionsToken corresponding to a passed in training
   * data source. If the user does not have an entitlement allowing them to
   * train on the data source, this will throw an error. If licensing is
   * disabled, this will always succeed. To prevent unauthorized API use, you
   * should only use this access token to train models with the passed in
   * training file.
   */
  explicit TrainPermissionsToken(const dataset::DataSourcePtr& source) {
    entitlements().verifyDataSource(source);
  }
};

}  // namespace thirdai::licensing