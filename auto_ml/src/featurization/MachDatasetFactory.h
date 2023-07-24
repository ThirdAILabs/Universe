#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <auto_ml/src/Aliases.h>
#include <data/src/ColumnMap.h>
#include <data/src/Loader.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>

namespace thirdai::automl {

using bolt::nn::tensor::TensorList;

/**
 * Questions:
 *  1. Do we need fast approximation in introduce docs
 *  2. Do we need to wrap the CSV data source before and after cold start?
 *      - I think this is ok because we don't write back to a file after cold
 *        start, it will go straight to the tokenizer.
 */

class MachDatasetFactory {
 public:
  thirdai::data::LoaderPtr getDataLoader(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      size_t max_batches, bool include_mach_labels,
      dataset::DatasetShuffleConfig shuffle_config, bool verbose);

  thirdai::data::LoaderPtr getColdStartDataLoader(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      size_t max_batches, const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      dataset::DatasetShuffleConfig shuffle_config, bool verbose);

  TensorList featurizeInput(const MapInput& sample);

  TensorList featurizeInputBatch(const MapInputBatch& samples);

  std::pair<TensorList, TensorList> featurizeTrainingBatch(
      const MapInputBatch& samples, bool prehashed);

 private:
  thirdai::data::ColdStartTextAugmentationPtr _cold_start;
  thirdai::data::TransformationPtr _input_transformation;
  thirdai::data::TransformationPtr _mach_label_transformation;
  thirdai::data::TransformationPtr _entity_id_transformation;
  thirdai::data::TransformationPtr _prehashed_label_transformation;

  char _delimiter;
  std::string _input_column;
  std::string _entity_id_column;
  std::string _mach_label_column;

  thirdai::data::StatePtr _state;
};

}  // namespace thirdai::automl