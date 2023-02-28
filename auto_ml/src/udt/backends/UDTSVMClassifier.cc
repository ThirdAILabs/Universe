#include "UDTSVMClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/DatasetLoaderWrappers.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/stl.h>
#include <stdexcept>

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

dataset::DatasetLoaderPtr getSVMDatasetLoader(
    const dataset::DataSourcePtr& source, bool shuffle) {
  return std::make_unique<dataset::DatasetLoader>(
      source,
      std::make_shared<dataset::SvmFeaturizer>(
          /* softmax_for_multiclass = */ true),
      shuffle);
}

void UDTSVMClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  _binary_prediction_threshold = utils::trainClassifier(
      data, learning_rate, epochs, validation, batch_size_opt,
      max_in_memory_batches, metrics, callbacks, verbose, logging_interval,
      &getSVMDatasetLoader, _model);
}

py::object UDTSVMClassifier::evaluate(const dataset::DataSourcePtr& data,
                                      const std::vector<std::string>& metrics,
                                      bool sparse_inference,
                                      bool return_predicted_class, bool verbose,
                                      bool return_metrics) {
  auto dataset_loader = getSVMDatasetLoader(data, /* shuffle = */ false);
  return utils::evaluate(metrics, sparse_inference, return_predicted_class,
                         verbose, return_metrics, _model, dataset_loader,
                         _binary_prediction_threshold);
}

py::object UDTSVMClassifier::predict(const MapInput& sample,
                                     bool sparse_inference,
                                     bool return_predicted_class) {
  BoltVector output = _model->predictSingle(
      {dataset::SvmDatasetLoader::toSparseVector(sample)}, sparse_inference);

  if (return_predicted_class) {
    return py::cast(
        utils::predictedClass(output, _binary_prediction_threshold));
  }

  return utils::convertBoltVectorToNumpy(output);
}

py::object UDTSVMClassifier::predictBatch(const MapInputBatch& samples,
                                          bool sparse_inference,
                                          bool return_predicted_class) {
  BoltBatch outputs = _model->predictSingleBatch(
      {dataset::SvmDatasetLoader::toSparseVectors(samples)}, sparse_inference);

  if (return_predicted_class) {
    return utils::predictedClasses(outputs, _binary_prediction_threshold);
  }

  return utils::convertBoltBatchToNumpy(outputs);
}

template void UDTSVMClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTSVMClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTSVMClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _model, _freeze_hash_tables,
          _binary_prediction_threshold);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTSVMClassifier)