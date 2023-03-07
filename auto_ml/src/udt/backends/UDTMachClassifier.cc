#include "UDTMachClassifier.h"
#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>

namespace thirdai::automl::udt {

UDTMachClassifier::UDTMachClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_name, data::CategoricalDataTypePtr target,
    uint32_t n_target_classes, bool integer_target,
    const data::TabularOptions& tabular_options,
    const config::ArgumentMap& user_args) {
  uint32_t hidden_dim = user_args.get<uint32_t>(
      "embedding_dimension", "integer", defaults::HIDDEN_DIM);
  _model = utils::defaultModel(tabular_options.feature_hash_range, hidden_dim,
                               output_dim,
                               /* use_sigmoid_bce = */ true);

  // TODO(david) move things like label block and coldstart out of here and into
  // a classifier utils file?
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
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, metrics, callbacks, verbose,
      logging_interval, _dataset_factory);

  auto train_dataset =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  utils::train(_model, train_dataset, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables,
               licensing::TrainPermissionsToken(data));
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       bool verbose, bool return_metrics) {
                                        // should this just return an ordered list 
                                       }

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class) {}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& sample,
                                           bool sparse_inference,
                                           bool return_predicted_class) {}

std::vector<dataset::Explanation> UDTMachClassifier::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {}

void UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<Validation>& validation,
    const std::vector<bolt::CallbackPtr>& callbacks, bool verbose) {}

py::object UDTMachClassifier::embedding(const MapInput& sample) {}

py::object UDTMachClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {}

std::string UDTMachClassifier::className(uint32_t class_id) const {}

data::TabularDatasetFactoryPtr UDTMachClassifier::tabularDatasetFactory()
    const {}

dataset::CategoricalBlockPtr UDTMachClassifier::labelBlock(
    const std::string& target_name, data::CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target) {}

template <class Archive>
void serialize(Archive& archive);

bolt::BoltGraphPtr _model;
bool _freeze_hash_tables;

}  // namespace thirdai::automl::udt