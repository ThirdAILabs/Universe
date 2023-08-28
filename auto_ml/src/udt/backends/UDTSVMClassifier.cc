#include "UDTSVMClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <dataset/src/DatasetLoaderWrappers.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/stl.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <stdexcept>

namespace thirdai::automl::udt {

UDTSVMClassifier::UDTSVMClassifier(
    uint32_t n_target_classes, uint32_t input_dim,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _classifier(utils::Classifier::make(
          utils::buildModel(
              /* input_dim= */ input_dim,
              /* output_dim= */ n_target_classes,
              /* args= */ user_args, /* model_config= */ model_config),
          user_args.get<bool>("freeze_hash_tables", "boolean",
                              defaults::FREEZE_HASH_TABLES))) {}

UDTSVMClassifier::UDTSVMClassifier(const proto::udt::UDTSvmClassifier& svm)
    : _classifier(utils::Classifier::fromProto(svm.classifier())) {}

py::object UDTSVMClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  auto featurizer = std::make_shared<dataset::SvmFeaturizer>();
  auto train_dataset_loader = svmDatasetLoader(
      data, /* shuffle= */ true, /* shuffle_config= */ options.shuffle_config);

  dataset::DatasetLoaderPtr val_dataset_loader;
  if (val_data) {
    val_dataset_loader = svmDatasetLoader(val_data, /* shuffle= */ false);
  }

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            train_metrics, val_dataset_loader, val_metrics,
                            callbacks, options, comm);
}

py::object UDTSVMClassifier::evaluate(const dataset::DataSourcePtr& data,
                                      const std::vector<std::string>& metrics,
                                      bool sparse_inference, bool verbose,
                                      std::optional<uint32_t> top_k) {
  (void)top_k;

  auto dataset = svmDatasetLoader(data, /* shuffle= */ false);

  return _classifier->evaluate(dataset, metrics, sparse_inference, verbose);
}

py::object UDTSVMClassifier::predict(const MapInput& sample,
                                     bool sparse_inference,
                                     bool return_predicted_class,
                                     std::optional<uint32_t> top_k) {
  auto inputs =
      bolt::convertVectors({dataset::SvmDatasetLoader::toSparseVector(sample)},
                           _classifier->model()->inputDims());
  return _classifier->predict(inputs, sparse_inference, return_predicted_class,
                              /* single= */ true, top_k);
}

py::object UDTSVMClassifier::predictBatch(const MapInputBatch& samples,
                                          bool sparse_inference,
                                          bool return_predicted_class,
                                          std::optional<uint32_t> top_k) {
  auto inputs =
      bolt::convertBatch({dataset::SvmDatasetLoader::toSparseVectors(samples)},
                         _classifier->model()->inputDims());
  return _classifier->predict(inputs, sparse_inference, return_predicted_class,
                              /* single= */ false, top_k);
}

proto::udt::UDT* UDTSVMClassifier::toProto(bool with_optimizer) const {
  auto* udt = new proto::udt::UDT();

  auto* svm = udt->mutable_svm();

  svm->set_allocated_classifier(_classifier->toProto(with_optimizer));

  return udt;
}

template void UDTSVMClassifier::serialize(cereal::BinaryInputArchive&,
                                          const uint32_t version);
template void UDTSVMClassifier::serialize(cereal::BinaryOutputArchive&,
                                          const uint32_t version);

template <class Archive>
void UDTSVMClassifier::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_SVM_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_SVM_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_SVM_CLASSIFIER_VERSION after serialization
  // changes
  archive(cereal::base_class<UDTBackend>(this), _classifier);
}

dataset::DatasetLoaderPtr UDTSVMClassifier::svmDatasetLoader(
    dataset::DataSourcePtr data_source, bool shuffle,
    dataset::DatasetShuffleConfig shuffle_config) {
  auto featurizer = std::make_shared<dataset::SvmFeaturizer>();

  auto dataset_loader = std::make_unique<dataset::DatasetLoader>(
      std::move(data_source), featurizer, shuffle, shuffle_config);

  return dataset_loader;
}
}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTSVMClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTSVMClassifier,
                     thirdai::versions::UDT_SVM_CLASSIFIER_VERSION)