#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/data_pipeline/ColumnMap.h>
#include <dataset/src/data_pipeline/Transformation.h>
#include <algorithm>

namespace thirdai::dataset {

class FeaturizationPipeline {
 public:
  FeaturizationPipeline(std::vector<TransformationPtr> transformations,
                        std::vector<std::string> output_column_names)
      : _transformations(std::move(transformations)),
        _output_column_names(std::move(output_column_names)) {}

  std::vector<BoltVector> featurize(ColumnMap& columns) {
    for (auto& transformation : _transformations) {
      transformation->apply(columns);
    }

    auto output_columns = columns.selectColumns(_output_column_names);

    std::vector<BoltVector> output_vectors;
    output_vectors.reserve(columns.numRows());

    bool is_dense = std::any_of(output_columns.begin(), output_columns.end(),
                                [](auto& column) { return column->isDense(); });

    for (uint64_t row_index = 0; row_index < columns.numRows(); row_index++) {
      if (is_dense) {
        SegmentedDenseFeatureVector vector;
        for (const auto& column : output_columns) {
          // Call addFeatureSegment
          column->appendRowToVector(vector, row_index);
        }
        output_vectors.push_back(vector.toBoltVector());
      } else {
        SegmentedSparseFeatureVector vector;
        for (const auto& column : output_columns) {
          // Call addFeatureSegment
          column->appendRowToVector(vector, row_index);
        }
        output_vectors.push_back(vector.toBoltVector());
      }
    }

    return output_vectors;
  }

 private:
  std::vector<TransformationPtr> _transformations;
  std::vector<std::string> _output_column_names;
};

}  // namespace thirdai::dataset