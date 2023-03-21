#pragma once

#include <exceptions/src/Exceptions.h>
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

  bool fullAccess() { return _entitlements.count(FULL_ACCESS_ENTITLEMENT); }

  void verifySaveLoad() {
    if (hasFullModelAccess() || _entitlements.count("SAVE_LOAD")) {
      return;
    }

    throw exceptions::LicenseCheckException(
        "Saving and loading of models is not authorized under this license.");
  }

  void verifyAllowedNumberOfTrainingSamples(
      uint64_t total_num_training_samples) {
    if (hasFullModelAccess()) {
      return;
    }

    (void)total_num_training_samples;
  }

 private:
  bool hasFullModelAccess() {
    return _entitlements.count(FULL_ACCESS_ENTITLEMENT) ||
           _entitlements.count(FULL_MODEL_ENTITLEMENT);
  }

  bool hasFullDatasetAccess() {
    return _entitlements.count(FULL_ACCESS_ENTITLEMENT) ||
           _entitlements.count(FULL_DATASET_ENTITLEMENT);
  }

  std::unordered_set<std::string> _entitlements;
};

}  // namespace thirdai::licensing