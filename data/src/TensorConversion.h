#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

enum ValueFillType {
  Ones,
  SumToOne,
};

class IndexValueColumn {
 public:
  explicit IndexValueColumn(std::string indices,
                            ValueFillType value_fill_type = ValueFillType::Ones)
      : _indices(std::move(indices)), _value_fill_type(value_fill_type) {}

  IndexValueColumn(std::string indices, std::string values)
      : _indices(std::move(indices)), _values(std::move(values)) {}

  const auto& indices() const { return _indices; }

  const auto& values() const { return _values; }

  ValueFillType valueFillType() const { return _value_fill_type; }

 private:
  std::string _indices;
  std::optional<std::string> _values;
  ValueFillType _value_fill_type;
};

using IndexValueColumnList = std::vector<IndexValueColumn>;

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const IndexValueColumnList& columns_to_convert,
    size_t batch_size);

bolt::TensorList toTensors(const ColumnMap& columns,
                           const IndexValueColumnList& columns_to_convert);

}  // namespace thirdai::data