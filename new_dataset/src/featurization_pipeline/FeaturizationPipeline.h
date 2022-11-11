#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <algorithm>
#include <memory>

namespace thirdai::dataset {

class FeaturizationPipeline;
using FeaturizationPipelinePtr = std::shared_ptr<FeaturizationPipeline>;

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

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static FeaturizationPipelinePtr load(const std::string& filename);

  static FeaturizationPipelinePtr load_stream(std::istream& input_stream);

 private:
  std::vector<TransformationPtr> _transformations;

  // Private constructor for cereal.
  FeaturizationPipeline(){};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::dataset
