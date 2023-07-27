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

class MachDatasetFactory {
 public:
  MachDatasetFactory() {}

  MachDatasetFactory(data::ColumnDataTypes input_data_types,
                     dataset::mach::MachIndexPtr mach_index,
                     const data::TabularOptions& options);

  thirdai::data::LoaderPtr getDataLoader(
      const dataset::DataSourcePtr& data_source, bool include_mach_labels,
      size_t batch_size, bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  thirdai::data::LoaderPtr getColdStartDataLoader(
      const dataset::DataSourcePtr& data_source,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation, bool include_mach_labels, size_t batch_size,
      bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  TensorList featurizeInput(const MapInput& sample);

  TensorList featurizeInputBatch(const MapInputBatch& samples);

  TensorList featurizeInputColdStart(
      MapInput sample, const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names);

  std::pair<TensorList, TensorList> featurizeTrainingBatch(
      const MapInputBatch& samples, bool prehashed);

  thirdai::data::ColumnMap applyInputTransformation(
      thirdai::data::ColumnMap columns) const {
    return _input_featurization->apply(std::move(columns), *_state);
  }

  thirdai::data::TransformationPtr createColdStartTransformation(
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      bool fast_approximation = false) const;

  const auto& machIndex() const { return _state->machIndex(); }

  void setIndex(dataset::mach::MachIndexPtr new_index) {
    _state->setMachIndex(std::move(new_index));
  }

  const std::string& machLabelColumnName() const {
    return _mach_label_column_name;
  }

  const std::string& entityIdColumnName() const {
    return _entity_id_column_name;
  }

  const data::ColumnDataTypes& dataTypes() const { return _input_data_types; }

 private:
  thirdai::data::LoaderPtr getDataLoaderHelper(
      const dataset::DataSourcePtr& data_source, bool include_mach_labels,
      size_t batch_size, bool shuffle, bool verbose,
      dataset::DatasetShuffleConfig shuffle_config,
      thirdai::data::TransformationPtr cold_start_transformation);

  std::string uniqueColumnName(std::string name) const;

  thirdai::data::TransformationPtr _input_featurization;
  thirdai::data::TransformationPtr _strings_to_entity_ids;
  thirdai::data::TransformationPtr _entity_ids_to_mach_labels;
  thirdai::data::TransformationPtr _strings_to_prehashed_mach_labels;

  data::ColumnDataTypes _input_data_types;
  char _delimiter;

  std::string _featurized_input_column_name;
  std::string _entity_id_column_name;
  std::string _mach_label_column_name;
  std::string _input_text_column_name;
  std::string _input_label_column_name;

  thirdai::data::StatePtr _state;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::automl