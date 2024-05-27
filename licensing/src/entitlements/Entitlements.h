#pragma once

#include "RestrictionTree.h"
#include <dataset/src/DataSource.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>
#include <exceptions/src/Exceptions.h>
#include <unordered_set>
#include <utility>

namespace thirdai::licensing {

class Entitlements {
 public:
  explicit Entitlements(const std::unordered_set<std::string>& entitlements)
      : _entitlements(RestrictionTree(entitlements)){};

  bool hasFullAccess() const { return !_entitlements.restrictions.has_value(); }

  void verifyFullAccess() const;

  void verifySaveLoad() const;

  void verifyAllowedNumberOfTrainingSamples(
      uint64_t total_num_training_samples) const;

  void verifyAllowedOutputDim(uint64_t output_dim) const;

  void verifyDataSource(const dataset::DataSourcePtr& source) const;

  void verifyNoDataSourceRetrictions() const;

 private:
  std::optional<ModelRestrictions> getModelRestrictions() const;

  std::optional<DatasetRestrictions> getDatasetRestrictions() const;

  RestrictionTree _entitlements;
};

}  // namespace thirdai::licensing