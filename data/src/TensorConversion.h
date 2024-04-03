#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>

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

ar::ConstArchivePtr outputColumnsToArchive(
    const OutputColumnsList& output_columns);

OutputColumnsList outputColumnsFromArchive(const ar::Archive& archive);

std::vector<bolt::TensorList> toTensorBatches(
    const ColumnMap& columns, const OutputColumnsList& columns_to_convert,
    size_t batch_size);

bolt::TensorList toTensors(const ColumnMap& columns,
                           const OutputColumnsList& columns_to_convert);

bolt::LabeledDataset toLabeledDataset(const ColumnMap& columns,
                                      const OutputColumnsList& input_columns,
                                      const OutputColumnsList& label_columns,
                                      size_t batch_size);

}  // namespace thirdai::data