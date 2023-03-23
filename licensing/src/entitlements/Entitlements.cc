#include "Entitlements.h"

namespace thirdai::licensing {

void Entitlements::verifySaveLoad() {
  if (hasFullModelAccess()) {
    return;
  }

  FinegrainedModelAccess model_access = getFinegrainedModelAccess();

  if (model_access.load_save) {
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
    uint64_t total_num_training_samples) {
  if (hasFullModelAccess()) {
    return;
  }

  FinegrainedModelAccess model_access = getFinegrainedModelAccess();

  if (total_num_training_samples < model_access.max_train_samples) {
    return;
  }

  throw exceptions::LicenseCheckException(
      "This model has exceeded the number of training examples allowed for "
      "this license.");
}

void Entitlements::verifyAllowedOutputDim(uint64_t output_dim) {
  if (hasFullModelAccess()) {
    return;
  }

  FinegrainedModelAccess model_access = getFinegrainedModelAccess();

  if (output_dim < model_access.max_output_dim) {
    return;
  }

  throw exceptions::LicenseCheckException(
      "This model's output dim is too large to be allowed under this "
      "license.");
}

void Entitlements::verifyDataSource(const dataset::DataSourcePtr& source) {
  if (hasFullDatasetAccess()) {
    return;
  }

  FinegrainedDatasetAccess dataset_access = getFinegrainedDatasetAccess();

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

  if (!dataset_access.dataset_hashes.count(sha256File(file_path))) {
    throw exceptions::LicenseCheckException(
        "This dataset is not authorized under this license.");
  }
}

bool Entitlements::hasFullModelAccess() {
  return hasFullAccess() ||
         std::holds_alternative<FullModelAccess>(
             std::get<FinegrainedFullAccess>(_entitlements.access)
                 .model_access);
}

bool Entitlements::hasFullDatasetAccess() {
  return hasFullAccess() ||
         std::holds_alternative<FullDatasetAccess>(
             std::get<FinegrainedFullAccess>(_entitlements.access)
                 .dataset_access);
}

FinegrainedModelAccess Entitlements::getFinegrainedModelAccess() {
  return std::get<FinegrainedModelAccess>(
      std::get<FinegrainedFullAccess>(_entitlements.access).model_access);
}

FinegrainedDatasetAccess Entitlements::getFinegrainedDatasetAccess() {
  return std::get<FinegrainedDatasetAccess>(
      std::get<FinegrainedFullAccess>(_entitlements.access).dataset_access);
}

}  // namespace thirdai::licensing