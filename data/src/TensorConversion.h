#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

enum ValueFillType {
  Ones,
  SumToOne,
};

class OutputColumns {
 public:
  static OutputColumns sparse(
      std::string indices,
      ValueFillType value_fill_type = ValueFillType::Ones) {
    return OutputColumns(indices, std::nullopt, value_fill_type);
  }

  static OutputColumns sparse(std::string indices, std::string values) {
    return OutputColumns(indices, values, ValueFillType::Ones);
  }

  static OutputColumns dense(std::string values) {
    return OutputColumns(std::nullopt, values, ValueFillType::Ones);
  }

  const auto& indices() const { return _indices; }

  const auto& values() const { return _values; }

  ValueFillType valueFillType() const { return _value_fill_type; }

  OutputColumns() {}  // For cereal

 private:
  OutputColumns(std::optional<std::string> indices,
                std::optional<std::string> values,
                ValueFillType value_fill_type)
      : _indices(std::move(indices)),
        _values(std::move(values)),
        _value_fill_type(value_fill_type) {}

  std::optional<std::string> _indices;
  std::optional<std::string> _values;
  ValueFillType _value_fill_type;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_indices, _values, _value_fill_type);
  }
};

using OutputColumnsList = std::vector<OutputColumns>;

ar::ConstArchivePtr outputColumnsToArchive(
    const OutputColumnsList& output_columns);

OutputColumnsList outputColumnsFromArchive(const ar::Archive& archive);

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const OutputColumnsList& columns_to_convert,
    size_t batch_size);

bolt::TensorList toTensors(const ColumnMap& columns,
                           const OutputColumnsList& columns_to_convert);

}  // namespace thirdai::data