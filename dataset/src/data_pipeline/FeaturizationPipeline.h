#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/data_pipeline/ColumnMap.h>
#include <dataset/src/data_pipeline/Transformation.h>
#include <algorithm>

namespace thirdai::dataset {

class FeaturizationPipeline {
 public:
  explicit FeaturizationPipeline(std::vector<TransformationPtr> transformations)
      : _transformations(std::move(transformations)) {}

  ColumnMap featurize(ColumnMap columns) {
    for (auto& transformation : _transformations) {
      transformation->apply(columns);
    }

    return columns;
  }

 private:
  std::vector<TransformationPtr> _transformations;
};

}  // namespace thirdai::dataset