#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <data/src/ColumnMap.h>
#include <data/src/Loader.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachIndex.h>

namespace thirdai::automl {

using bolt::nn::tensor::TensorList;

/**
 * Question:
 *  Do we need to wrap the CSV data source before and after cold start?
 *      - I think this is ok because we don't write back to a file after cold
 *        start, it will go straight to the tokenizer.
 */

class MachDatasetFactory {
 public:
  MachDatasetFactory() {}

  MachDatasetFactory(const data::ColumnDataTypes& input_data_types,
                     dataset::mach::MachIndexPtr mach_index,
                     const data::TabularOptions& options);

  thirdai::data::LoaderPtr getDataLoader(
      const dataset::DataSourcePtr& data_source, bool include_mach_labels,
      dataset::DatasetShuffleConfig shuffle_config, bool verbose);

  thirdai::data::LoaderPtr getColdStartDataLoader(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool include_mach_labels, bool fast_approximation,
      dataset::DatasetShuffleConfig shuffle_config, bool verbose);

  TensorList featurizeInput(const MapInput& sample);

  TensorList featurizeInputBatch(const MapInputBatch& samples);

  TensorList featurizeInputColdStart(
      MapInput sample, const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names);

  std::pair<TensorList, TensorList> featurizeTrainingBatch(
      const MapInputBatch& samples, bool prehashed);

  thirdai::data::ColumnMap applyInputTransformation(
      thirdai::data::ColumnMap columns) const {
    return _input_transformation->apply(std::move(columns), *_state);
  }

  thirdai::data::TransformationPtr createColdStartTransformation(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation = false) const;

  const auto& machIndex() const { return _state->machIndex(); }

  void setIndex(dataset::mach::MachIndexPtr new_index) {
    _state->setMachIndex(std::move(new_index));
  }

  const std::string& featurizedMachLabelColumn() const {
    return _featurized_mach_label_column;
  }

  const std::string& featurizedEntityIdColumn() const {
    return _featurized_entity_id_column;
  }

 private:
  thirdai::data::LoaderPtr getDataLoaderHelper(
      const dataset::DataSourcePtr& data_source, bool include_mach_labels,
      dataset::DatasetShuffleConfig shuffle_config, bool verbose,
      thirdai::data::TransformationPtr cold_start_transformation);

  thirdai::data::TransformationPtr _input_transformation;
  thirdai::data::TransformationPtr _entity_id_transformation;
  thirdai::data::TransformationPtr _mach_label_transformation;
  thirdai::data::TransformationPtr _prehashed_label_transformation;

  char _delimiter;
  std::string _featurized_input_column;
  std::string _featurized_entity_id_column;
  std::string _featurized_mach_label_column;
  std::string _cold_start_text_column;
  std::string _cold_start_label_column;

  thirdai::data::StatePtr _state;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::automl