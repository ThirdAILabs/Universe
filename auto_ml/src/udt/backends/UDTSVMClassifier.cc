#include "UDTSVMClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/DatasetLoaderWrappers.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace thirdai::automl::udt {

UDTSVMClassifier::UDTSVMClassifier(
    uint32_t n_target_classes, uint32_t input_dim,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _classifier(utils::buildModel(
                      /* input_dim= */ input_dim,
                      /* output_dim= */ n_target_classes,
                      /* args= */ user_args, /* model_config= */ model_config),
                  user_args.get<bool>("freeze_hash_tables", "boolean",
                                      defaults::FREEZE_HASH_TABLES)) {}

py::object UDTSVMClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  auto featurizer = std::make_shared<dataset::SvmFeaturizer>();
  auto train_dataset_loader = svmDatasetLoader(data, /* shuffle= */ true);

  std::optional<ValidationDatasetLoader> validation_dataset_loader =
      std::nullopt;
  if (validation) {
    validation_dataset_loader = ValidationDatasetLoader(
        svmDatasetLoader(validation->first, /* shuffle= */ false),
        validation->second);
  }

  return _classifier.train(train_dataset_loader, learning_rate, epochs,
                           validation_dataset_loader, batch_size_opt,
                           max_in_memory_batches, metrics, callbacks, verbose,
                           logging_interval);
}

py::object UDTSVMClassifier::evaluate(const dataset::DataSourcePtr& data,
                                      const std::vector<std::string>& metrics,
                                      bool sparse_inference,
                                      bool return_predicted_class, bool verbose,
                                      bool return_metrics) {
  auto dataset = svmDatasetLoader(data, /* shuffle= */ false);

  return _classifier.evaluate(dataset, metrics, sparse_inference,
                              return_predicted_class, verbose, return_metrics);
}

py::object UDTSVMClassifier::predict(const MapInput& sample,
                                     bool sparse_inference,
                                     bool return_predicted_class) {
  return _classifier.predict(
      {dataset::SvmDatasetLoader::toSparseVector(sample)}, sparse_inference,
      return_predicted_class);
}

py::object UDTSVMClassifier::predictBatch(const MapInputBatch& samples,
                                          bool sparse_inference,
                                          bool return_predicted_class) {
  return _classifier.predictBatch(
      {dataset::SvmDatasetLoader::toSparseVectors(samples)}, sparse_inference,
      return_predicted_class);
}

template void UDTSVMClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTSVMClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTSVMClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _classifier);
}

dataset::DatasetLoaderPtr UDTSVMClassifier::svmDatasetLoader(
    dataset::DataSourcePtr data_source, bool shuffle) {
  auto featurizer = std::make_shared<dataset::SvmFeaturizer>();

  auto dataset_loader = std::make_unique<dataset::DatasetLoader>(
      std::move(data_source), featurizer, shuffle);

  return dataset_loader;
}
}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTSVMClassifier)