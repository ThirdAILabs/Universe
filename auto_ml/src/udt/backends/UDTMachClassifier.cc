#include "UDTMachClassifier.h"
#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/blocks/CategoricalMultiHashBlock.h>

namespace thirdai::automl::udt {

UDTMachClassifier::UDTMachClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_name, data::CategoricalDataTypePtr target,
    uint32_t n_target_classes, bool integer_target,
    const data::TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _classifier(
          utils::buildModel(
              /* input_dim= */ tabular_options.feature_hash_range,
              /* output_dim= */
              user_args.get<uint32_t>("mach_output_dim", "integer",
                                      autotuneMachOutputDim(n_target_classes)),
              /* args= */ user_args, /* model_config= */ model_config,
              /* use_sigmoid_bce = */ true),
          user_args.get<bool>("freeze_hash_tables", "boolean",
                              defaults::FREEZE_HASH_TABLES)) {
  // TODO(david) should we freeze hash tables for mach? how does this work with
  // coldstart? is this why we're getting bad msmarco accuracy?

  // TODO(david) move things like label block and coldstart out of here and into
  // a classifier utils file?

  uint32_t num_hashes

  _mach_index = dataset::MachIndex();

  _multi_hash_label_block =
      labelBlock(target_name, target, n_target_classes, integer_target);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{_multi_hash_label_block},
      std::set<std::string>{target_name}, tabular_options, force_parallel);
}

void UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  std::optional<ValidationDatasetLoader> validation_dataset_loader =
      std::nullopt;
  if (validation) {
    validation_dataset_loader =
        ValidationDatasetLoader(_dataset_factory->getDatasetLoader(
                                    validation->first, /* shuffle= */ false),
                                validation->second);
  }

  auto train_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  _classifier.train(train_dataset_loader, learning_rate, epochs,
                    validation_dataset_loader, batch_size_opt,
                    max_in_memory_batches, metrics, callbacks, verbose,
                    logging_interval, licensing::TrainPermissionsToken(data));
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       bool verbose, bool return_metrics) {
  auto dataset = _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  return _classifier.evaluate(dataset, metrics, sparse_inference,
                              return_predicted_class, verbose, return_metrics);
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class) {
  return _classifier.predict(_dataset_factory->featurizeInput(sample),
                             sparse_inference, return_predicted_class);
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class) {
  return _classifier.predictBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference,
      return_predicted_class);
}

void UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<ValidationDataSource>& validation,
    const std::vector<bolt::CallbackPtr>& callbacks, bool verbose) {}

py::object UDTMachClassifier::embedding(const MapInput& sample) {}

py::object UDTMachClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {}

std::string UDTMachClassifier::className(uint32_t class_id) const {}

dataset::CategoricalBlockPtr UDTMachClassifier::labelBlock(
    const std::string& target_name, data::CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target) {}

template <class Archive>
void serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _classifier,
          _multi_hash_label_block, _dataset_factory);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachlassifier)