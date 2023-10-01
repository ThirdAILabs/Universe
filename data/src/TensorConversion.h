#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/Transformation.h>
#include <cstddef>
#include <utility>

namespace thirdai::data {

enum ValueFillType {
  Ones,
  SumToOne,
};

class OutputColumns {
 public:
  explicit OutputColumns(std::string indices,
                         ValueFillType value_fill_type = ValueFillType::Ones)
      : _indices(std::move(indices)), _value_fill_type(value_fill_type) {}

  OutputColumns(std::string indices, std::string values)
      : _indices(std::move(indices)), _values(std::move(values)) {}

  const auto& indices() const { return _indices; }

  const auto& values() const { return _values; }

  ValueFillType valueFillType() const { return _value_fill_type; }

  OutputColumns() {}  // For cereal

 private:
  std::string _indices;
  std::optional<std::string> _values;
  ValueFillType _value_fill_type;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_indices, _values, _value_fill_type);
  }
};

using OutputColumnsList = std::vector<OutputColumns>;

struct TransformConfig {
  TransformConfig(
      thirdai::data::TransformationPtr _transform,
      thirdai::data::OutputColumnsList _input_columns,
      std::optional<thirdai::data::OutputColumnsList> _label_columns)
      : transform(std::move(_transform)),
        input_columns(std::move(_input_columns)),
        label_columns(std::move(_label_columns)) {}

  TransformConfig() {}

  thirdai::data::TransformationPtr transform;
  thirdai::data::OutputColumnsList input_columns;
  std::optional<thirdai::data::OutputColumnsList> label_columns;
};

struct TransformedTable {
  TransformedTable(thirdai::data::ColumnMap _table,
                   thirdai::data::OutputColumnsList _inputs,
                   std::optional<thirdai::data::OutputColumnsList> _labels)
      : table(std::move(_table)),
        inputs(std::move(_inputs)),
        labels(std::move(_labels)) {}

  TransformedTable(thirdai::data::ColumnMap table,
                   const TransformConfig& transform_config,
                   thirdai::data::State& state);

  void removeIntermediateColumns();

  thirdai::data::ColumnMap table;
  thirdai::data::OutputColumnsList inputs;
  std::optional<thirdai::data::OutputColumnsList> labels;
};

struct TransformedIterator {
  TransformedIterator(thirdai::data::ColumnMapIteratorPtr _iter,
                      thirdai::data::OutputColumnsList _inputs,
                      std::optional<thirdai::data::OutputColumnsList> _labels)
      : iter(std::move(_iter)),
        inputs(std::move(_inputs)),
        labels(std::move(_labels)) {}

  TransformedIterator(thirdai::data::ColumnMapIteratorPtr iter,
                      const TransformConfig& transform_config,
                      thirdai::data::StatePtr state);

  thirdai::data::ColumnMapIteratorPtr iter;
  thirdai::data::OutputColumnsList inputs;
  std::optional<thirdai::data::OutputColumnsList> labels;
};

struct TransformedTensors {
  explicit TransformedTensors(const TransformedTable& table);

  bolt::TensorList inputs;
  std::optional<bolt::TensorList> labels;
};

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const OutputColumnsList& columns_to_convert,
    size_t batch_size);

bolt::TensorList toTensors(const ColumnMap& columns,
                           const OutputColumnsList& columns_to_convert);

bolt::LabeledDataset toLabeledDataset(const TransformedTable&,
                                      size_t batch_size);

}  // namespace thirdai::data