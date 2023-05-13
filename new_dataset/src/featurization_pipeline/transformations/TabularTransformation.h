#pragma once

#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::data {

/**
 * To transition to the new data pipeline, we wrap tabular featurizer to conform
 * to Transformation interface.
 *
 * Since the new data pipeline operates on column maps, we need implementations
 * of ColumnarInputSample and ColumnarInputBatch for ColumnMap.
 */

using StringColumns = std::unordered_map<std::string, columns::StringColumnPtr>;

class ColumnMapSampleRef final : public dataset::ColumnarInputSample {
 public:
  explicit ColumnMapSampleRef(const data::ColumnMap& columns, size_t sample_id)
      : _columns(columns), _sample_id(sample_id) {}

  std::string column(const dataset::ColumnIdentifier& column) final {
    return (*_columns.getStringColumn(column.name()))[_sample_id];
  }

  uint32_t size() final { return _columns.columns().size(); }

 private:
  const data::ColumnMap& _columns;
  size_t _sample_id;
};

class ColumnMapBatchRef final : public dataset::ColumnarInputBatch {
 public:
  explicit ColumnMapBatchRef(const data::ColumnMap& columns) {
    _samples.reserve(columns.numRows());
    for (uint32_t sample_id = 0; sample_id < columns.numRows(); sample_id++) {
      _samples.push_back(ColumnMapSampleRef(columns, sample_id));
    }
  }

  dataset::ColumnarInputSample& at(uint32_t index) final {
    return _samples.at(index);
  }

  uint32_t size() const final { return _samples.size(); }

 private:
  std::vector<ColumnMapSampleRef> _samples;
};

/**
 * Wrapper for TabularFeaturizer that conforms to Transformation interface of
 * new data pipeline.
 */
class TabularTransformation final : public Transformation {
 public:
  TabularTransformation(dataset::TabularFeaturizerPtr&& featurizer,
                        std::vector<std::string> output_columns)
      : _featurizer(std::move(featurizer)),
        _output_columns(std::move(output_columns)) {
    if (_featurizer->getNumDatasets() != _output_columns.size()) {
      throw std::invalid_argument(
          "[TabularTransformation] Mismatched number of output columns (" +
          std::to_string(_output_columns.size()) +
          ")and number of returned datasets (" +
          std::to_string(_featurizer->getNumDatasets()) + ").");
    }
  }

  void apply(ColumnMap& columns) final {
    ColumnMapBatchRef columns_ref(columns);
    auto features = _featurizer->featurize(columns_ref);
    auto dimensions = _featurizer->getDimensions();
    for (uint32_t col_id = 0; col_id < _output_columns.size(); col_id++) {
      auto col_name = _output_columns[col_id];
      auto new_column = columns::CppBoltVectorColumn::make(
          /* data= */ std::move(features[col_id]),
          /* dim= */ dimensions[col_id]);
      columns.setColumn(col_name, std::move(new_column));
    }
  }

 private:
  dataset::TabularFeaturizerPtr _featurizer;
  std::vector<std::string> _output_columns;
};

}  // namespace thirdai::data