#pragma once

#include "EntitlementTree.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/Utils.h>
#include <unordered_set>
#include <utility>

namespace thirdai::licensing {

class Entitlements {
 public:
  explicit Entitlements(const std::unordered_set<std::string>& entitlements)
      : _entitlements(EntitlementTree(entitlements)){};

  bool hasFullAccess() const {
    return std::holds_alternative<FullAccess>(_entitlements.access);
  }

  void verifyFullAccess() const;

  void verifySaveLoad();

  void verifyAllowedNumberOfTrainingSamples(
      uint64_t total_num_training_samples);

  void verifyAllowedOutputDim(uint64_t output_dim);

  void verifyDataSource(const dataset::DataSourcePtr& source);

 private:
  bool hasFullModelAccess();

  bool hasFullDatasetAccess();

  // This will throw an exception if hasFullModelAccess() is true
  FinegrainedModelAccess getFinegrainedModelAccess();

  // This will throw an exception if hasFullDatasetAccess() is true
  FinegrainedDatasetAccess getFinegrainedDatasetAccess();

  EntitlementTree _entitlements;
};

}  // namespace thirdai::licensing