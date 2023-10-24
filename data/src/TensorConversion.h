#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <data/src/ColumnMap.h>
#include <proto/output_columns.pb.h>

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

  explicit OutputColumns(const proto::data::OutputColumns& output_columns);

  OutputColumns(std::string indices, std::string values)
      : _indices(std::move(indices)), _values(std::move(values)) {}

  const auto& indices() const { return _indices; }

  const auto& values() const { return _values; }

  ValueFillType valueFillType() const { return _value_fill_type; }

  proto::data::OutputColumns* toProto() const;

 private:
  std::string _indices;
  std::optional<std::string> _values;
  ValueFillType _value_fill_type;
};

using OutputColumnsList = std::vector<OutputColumns>;

proto::data::OutputColumnsList* outputColumnsListToProto(
    const OutputColumnsList& output_columns_list);

OutputColumnsList outputColumnsListFromProto(

    const proto::data::OutputColumnsList& output_columns_list);

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const OutputColumnsList& columns_to_convert,
    size_t batch_size);

bolt::TensorList toTensors(const ColumnMap& columns,
                           const OutputColumnsList& columns_to_convert);

}  // namespace thirdai::data