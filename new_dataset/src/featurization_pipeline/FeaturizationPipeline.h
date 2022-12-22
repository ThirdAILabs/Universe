#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <algorithm>
#include <memory>

namespace thirdai::data {

class FeaturizationPipeline;
using FeaturizationPipelinePtr = std::shared_ptr<FeaturizationPipeline>;

class FeaturizationPipeline {
 public:
  explicit FeaturizationPipeline(std::vector<TransformationPtr> transformations)
      : _transformations(std::move(transformations)) {}

  ColumnMap featurize(ColumnMap columns,
                      bool prepare_for_backpropagate = false) {
    for (auto& transformation : _transformations) {
      transformation->apply(columns, prepare_for_backpropagate);
    }

    return columns;
  }

  ContributionColumnMap explain(ColumnMap columns,
                                ContributionColumnMap contribution_columns) {
    for (auto& transformation : _transformations) {
      transformation->backpropagate(columns, contribution_columns);
    }
    return contribution_columns;
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

}  // namespace thirdai::data
