#include "TrainPermissionsToken.h"
#include <licensing/src/CheckLicense.h>

namespace thirdai::licensing {

TrainPermissionsToken::TrainPermissionsToken(
    const dataset::DataSourcePtr& training_source) {
  if (entitlements().fullAccess()) {
    return;
  }

  // If the user just has a demo license and we are going to read in the dataset
  // from the resourceName, we require FileDataSources or ColdStartDataSources.
  // This prevents a user from extending the DataSource class in python and
  // making resourceName() point to a valid file, while the actual nextLine call
  // returns lines from some other file they want to train on.
  if (!dynamic_cast<dataset::FileDataSource*>(training_source.get()) &&
      !dynamic_cast<dataset::cold_start::ColdStartDataSource*>(
          training_source.get())) {
    throw exceptions::LicenseCheckException(
        "Can only train on file data sources with a demo license");
  }

  std::string file_path = training_source->resourceName();

  if (!_entitlements.count(sha256File(file_path))) {
    throw exceptions::LicenseCheckException(
        "This dataset is not authorized under this license.");
  }
}

TrainPermissionsToken::TrainPermissionsToken() { assertUserHasFullAccess(); }

}  // namespace thirdai::licensing