#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/Utils.h>
#include <unordered_set>
#include <utility>

namespace thirdai::licensing {

const std::string FULL_ACCESS_ENTITLEMENT = "FULL_ACCESS";
const std::string FULL_MODEL_ENTITLEMENT = "FULL_MODEL";
const std::string FULL_DATASET_ENTITLEMENT = "FULL_DATASET";

class Entitlements {
 public:
  explicit Entitlements(std::unordered_set<std::string> entitlements)
      : _entitlements(std::move(entitlements)){};

  Entitlements(){};

  bool hasFullAccess() { return _entitlements.count(FULL_ACCESS_ENTITLEMENT); }

  void verifyFullAccess() {
    if (!hasFullAccess()) {
      throw exceptions::LicenseCheckException(
          "You must have a full license to perform this operation.");
    }
  }

  void verifySaveLoad() {
    if (fullModelAccess() || _entitlements.count("SAVE_LOAD")) {
      return;
    }

    throw exceptions::LicenseCheckException(
        "Saving and loading of models is not authorized under this license.");
  }

  void verifyAllowedNumberOfTrainingSamples(
      uint64_t total_num_training_samples) {
    if (fullModelAccess()) {
      return;
    }

    (void)total_num_training_samples;
  }

  void verifyAllowedOutputDim(uint64_t output_dim) {
    if (fullModelAccess()) {
      return;
    }

    (void)output_dim;
  }

  bool contains(const std::string& key) { return _entitlements.count(key); }

  void verifyDataSource(const dataset::DataSourcePtr& source) {
    if (fullDatasetAccess()) {
      return;
    }

    // If the user just has a demo license and we are going to read in the
    // dataset from the resourceName, we require FileDataSources or
    // ColdStartDataSources. This prevents a user from extending the DataSource
    // class in python and making resourceName() point to a valid file, while
    // the actual nextLine call returns lines from some other file they want to
    // train on.
    if (!dynamic_cast<dataset::FileDataSource*>(source.get()) &&
        !dynamic_cast<dataset::cold_start::ColdStartDataSource*>(
            source.get())) {
      throw exceptions::LicenseCheckException(
          "Can only train on file data sources with this license");
    }

    std::string file_path = source->resourceName();

    if (!_entitlements.count(sha256File(file_path))) {
      throw exceptions::LicenseCheckException(
          "This dataset is not authorized under this license.");
    }
  }

 private:
  bool fullModelAccess() {
    return hasFullAccess() || _entitlements.count(FULL_MODEL_ENTITLEMENT);
  }

  bool fullDatasetAccess() {
    return hasFullAccess() || _entitlements.count(FULL_DATASET_ENTITLEMENT);
  }

  std::unordered_set<std::string> _entitlements;
};

}  // namespace thirdai::licensing