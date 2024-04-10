#include "Entitlements.h"
#include <licensing/src/entitlements/RestrictionTree.h>

#ifdef THIRDAI_BUILD_LICENSE
#include <licensing/src/Utils.h>
#endif

namespace thirdai::licensing {

void Entitlements::verifySaveLoad() const {
  std::optional<ModelRestrictions> model_restrictions = getModelRestrictions();
  if (!model_restrictions) {
    return;
  }

  if (model_restrictions->can_load_save) {
    return;
  }

  throw exceptions::LicenseCheckException(
      "Saving and loading of models is not authorized under this license.");
}

void Entitlements::verifyFullAccess() const {
  if (!hasFullAccess()) {
    throw exceptions::LicenseCheckException(
        "You must have a full license to perform this operation.");
  }
}

void Entitlements::verifyAllowedNumberOfTrainingSamples(
    uint64_t total_num_training_samples) const {
  std::optional<ModelRestrictions> model_restrictions = getModelRestrictions();
  if (!model_restrictions) {
    return;
  }

  if (total_num_training_samples < model_restrictions->max_train_samples) {
    return;
  }

  throw exceptions::LicenseCheckException(
      "This model has exceeded the number of training examples allowed for "
      "this license.");
}

void Entitlements::verifyAllowedOutputDim(uint64_t output_dim) const {
  std::optional<ModelRestrictions> model_restrictions = getModelRestrictions();
  if (!model_restrictions) {
    return;
  }

  if (output_dim < model_restrictions->max_output_dim) {
    return;
  }

  throw exceptions::LicenseCheckException(
      "This model's output dim is too large to be allowed under this "
      "license.");
}

void Entitlements::verifyDataSource(
    const dataset::DataSourcePtr& source) const {
  std::optional<DatasetRestrictions> dataset_restrictions =
      getDatasetRestrictions();
  if (!dataset_restrictions) {
    return;
  }

  // If the user just has a demo license and we are going to read in the
  // dataset from the resourceName, we require FileDataSources or
  // ColdStartDataSources. This prevents a user from extending the DataSource
  // class in python and making resourceName() point to a valid file, while
  // the actual nextLine call returns lines from some other file they want to
  // train on.
  if (!dynamic_cast<dataset::FileDataSource*>(source.get()) &&
      !dynamic_cast<dataset::cold_start::ColdStartDataSource*>(source.get())) {
    throw exceptions::LicenseCheckException(
        "Can only train on file data sources with this license");
  }

  std::string file_path = source->resourceName();

  // We need this ifdef here because sha256File is only defined when
  // THIRDAI_BUILD_LICENSE is true (and when it is false we should never get to
  // this line anyways, since we should always have a FULL_ACCESS license)
#ifdef THIRDAI_BUILD_LICENSE
  if (!dataset_restrictions->datasetAllowed(
          /* dataset_hash = */ sha256File(file_path))) {
    throw exceptions::LicenseCheckException(
        "This dataset is not authorized under this license.");
  }
#endif
}

void Entitlements::verifyNoDataSourceRetrictions() const {
  if (getDatasetRestrictions()) {
#ifdef THIRDAI_BUILD_LICENSE
    throw exceptions::LicenseCheckException(
        "This dataset is not authorized under this license.");
#endif
  }
}

std::optional<ModelRestrictions> Entitlements::getModelRestrictions() const {
  if (!_entitlements.restrictions) {
    return std::nullopt;
  }

  return _entitlements.restrictions->model_restrictions;
}

std::optional<DatasetRestrictions> Entitlements::getDatasetRestrictions()
    const {
  if (!_entitlements.restrictions) {
    return std::nullopt;
  }

  return _entitlements.restrictions->dataset_restrictions;
}

}  // namespace thirdai::licensing