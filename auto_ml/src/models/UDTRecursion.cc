#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <auto_ml/src/models/UDTRecursion.h>
#include <dataset/src/RecursionWrapper.h>
#include <memory>
#include <string>

namespace thirdai::automl::models {

UDTRecursion::UDTRecursion(data::ColumnDataTypes data_types,
                           std::string target_column, char column_delimiter)
    : _target_sequence(data::asSequence(data_types.at(target_column))),
      _udt_data_types(std::move(data_types)),
      _target_column(std::move(target_column)),
      _column_delimiter(column_delimiter) {
  if (_target_sequence) {
    for (uint32_t step = 1; step < _target_sequence->length; step++) {
      _intermediate_column_names.push_back(_target_column + "_" +
                                           std::to_string(step));
    }
    for (const auto& column_name : _intermediate_column_names) {
      _udt_data_types[column_name] =
          std::make_shared<data::CategoricalDataType>();
    }
    _udt_data_types[_target_column] =
        std::make_shared<data::CategoricalDataType>();
  }
}

data::ColumnDataTypes UDTRecursion::modifiedDataTypes() const {
  return _udt_data_types;
}

dataset::DataSourcePtr UDTRecursion::wrapDataSource(
    dataset::DataSourcePtr source) const {
  auto target_batch_size = source->getMaxBatchSize();
  return dataset::RecursionWrapper::make(
      /* source= */ std::move(source),
      /* column_delimiter= */ _column_delimiter,
      /* sequence_delimiter= */ _target_sequence->delimiter,
      /* sequence_column_name= */ _target_column,
      /* intermediate_column_names= */ _intermediate_column_names,
      /* target_batch_size= */ target_batch_size);
}

}  // namespace thirdai::automl::models