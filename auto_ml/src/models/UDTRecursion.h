#pragma once

#include <cereal/access.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <dataset/src/DataSource.h>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl::models {

class UDTRecursion {
 public:
  UDTRecursion(data::ColumnDataTypes data_types, std::string target_column,
               char column_delimiter)
      : _target_sequence(data::asSequence(data_types.at(target_column))),
        _udt_data_types(std::move(data_types)),
        _target_column(std::move(target_column)),
        _column_delimiter(column_delimiter) {
    if (_target_sequence) {
      _intermediate_column_names = intermediateColumnNames();
      expandTargetSequenceIntoCategoricalColumns();
    }

    // Here, the target sequence type has been converted into categorical
    // columns. Remaining sequences in the data types map must be input columns.
    for (const auto& [column_name, column_type] : _udt_data_types) {
      if (data::asSequence(column_type)) {
        throw std::invalid_argument(
            "The sequence data type can only be used for targets columns. For "
            "input columns, please use categorical, text, numerical, or date "
            "instead.");
      }
    }
  }

  UDTRecursion() {}

  bool targetIsRecursive() const { return !!_target_sequence; }

  uint32_t depth() const { return _target_sequence->length; }

  data::ColumnDataTypes modifiedDataTypes() const;

  dataset::DataSourcePtr wrapDataSource(
      dataset::DataSourcePtr data_source) const;

  template <typename PredictFn>
  void callPredictRecursively(MapInput sample, PredictFn predict) const {
    for (const auto& column_name : _intermediate_column_names) {
      sample[column_name] = predict(sample);
    }
  }

  template <typename PredictBatchFn, typename PredictionGetterFn>
  void callPredictBatchRecursively(
      MapInputBatch samples, PredictBatchFn predict_batch,
      PredictionGetterFn get_ith_prediction) const {
    for (const auto& column_name : _intermediate_column_names) {
      auto predictions = predict_batch(samples);
      for (uint32_t i = 0; i < samples.size(); i++) {
        samples[i][column_name] = get_ith_prediction(predictions, i);
      }
    }
  }

 private:
  std::vector<std::string> intermediateColumnNames() const;

  void expandTargetSequenceIntoCategoricalColumns();

  data::SequenceDataTypePtr _target_sequence;
  data::ColumnDataTypes _udt_data_types;
  std::string _target_column;
  char _column_delimiter;
  std::vector<std::string> _intermediate_column_names;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_target_sequence, _udt_data_types, _target_column,
            _column_delimiter, _intermediate_column_names);
  }
};

}  // namespace thirdai::automl::models