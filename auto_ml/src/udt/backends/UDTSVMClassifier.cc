#include "UDTSVMClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/graph/ExecutionConfig.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/DatasetLoaderWrappers.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <exceptions/src/Exceptions.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <variant>

namespace thirdai::automl::udt {

UDTSVMClassifier::UDTSVMClassifier(
    uint32_t n_target_classes, uint32_t input_dim,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args) {
  if (model_config) {
    _model = utils::loadModel({input_dim}, n_target_classes, *model_config);
  } else {
    uint32_t hidden_dim = user_args.get<uint32_t>("embedding_dim", "integer",
                                                  defaults::HIDDEN_DIM);
    _model = utils::defaultModel(input_dim, hidden_dim, n_target_classes);
  }

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

void UDTSVMClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  if (max_in_memory_batches.has_value()) {
    throw exceptions::NotImplemented(
        "Cannot yet train on an SVM dataset in a streaming fashion");
  }

  if (validation.has_value()) {
    throw exceptions::NotImplemented(
        "Cannot yet train on an SVM dataset with validation data");
  }

  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  auto [train_dataset, labels] =
      dataset::SvmDatasetLoader::loadDataset(data, batch_size);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, metrics, callbacks, verbose, logging_interval);

  utils::trainInMemory(_model, {{train_dataset}, labels}, train_config,
                       /* freeze_hash_tables= */ _freeze_hash_tables);
}

py::object UDTSVMClassifier::evaluate(const dataset::DataSourcePtr& data,
                                      const std::vector<std::string>& metrics,
                                      bool sparse_inference,
                                      bool return_predicted_class, bool verbose,
                                      bool return_metrics) {
  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      dataset::SvmDatasetLoader::loadDataset(data, defaults::BATCH_SIZE);

  auto [output_metrics, output] =
      _model->evaluate({test_data}, test_labels, eval_config);
  if (return_metrics) {
    return py::cast(output_metrics);
  }

  if (return_predicted_class) {
    return utils::predictedClasses(output);
  }

  return utils::convertInferenceTrackerToNumpy(output);
}

py::object UDTSVMClassifier::predict(const MapInput& sample,
                                     bool sparse_inference,
                                     bool return_predicted_class) {
  BoltVector output = _model->predictSingle(
      {dataset::SvmDatasetLoader::toSparseVector(sample)}, sparse_inference);

  if (return_predicted_class) {
    return py::cast(utils::predictedClass(output));
  }

  return utils::convertBoltVectorToNumpy(output);
}

py::object UDTSVMClassifier::predictBatch(const MapInputBatch& samples,
                                          bool sparse_inference,
                                          bool return_predicted_class) {
  BoltBatch outputs = _model->predictSingleBatch(
      {dataset::SvmDatasetLoader::toSparseVectors(samples)}, sparse_inference);

  if (return_predicted_class) {
    return utils::predictedClasses(outputs);
  }

  return utils::convertBoltBatchToNumpy(outputs);
}

template void UDTSVMClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTSVMClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTSVMClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _model, _freeze_hash_tables);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTSVMClassifier)