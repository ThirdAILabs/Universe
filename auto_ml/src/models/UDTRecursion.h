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
               char column_delimiter);

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