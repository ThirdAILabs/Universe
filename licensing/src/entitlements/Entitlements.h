#pragma once

#include "EntitlementTree.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
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

  void verifySaveLoad() const;

  void verifyAllowedNumberOfTrainingSamples(
      uint64_t total_num_training_samples) const;

  void verifyAllowedOutputDim(uint64_t output_dim) const;

  void verifyDataSource(const dataset::DataSourcePtr& source) const;

 private:
  bool hasFullModelAccess() const;

  bool hasFullDatasetAccess() const;

  // This will throw an exception if hasFullModelAccess() is true
  FinegrainedModelAccess getFinegrainedModelAccess() const;

  // This will throw an exception if hasFullDatasetAccess() is true
  FinegrainedDatasetAccess getFinegrainedDatasetAccess() const;

  EntitlementTree _entitlements;
};

}  // namespace thirdai::licensing