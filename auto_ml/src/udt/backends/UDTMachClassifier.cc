#include "UDTMachClassifier.h"
#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/blocks/MachBlocks.h>

namespace thirdai::automl::udt {

UDTMachClassifier::UDTMachClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_name,
    const data::CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target,
    const data::TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args) {
  uint32_t output_range = user_args.get<uint32_t>(
      "mach_output_dim", "integer", autotuneMachOutputDim(n_target_classes));
  uint32_t num_hashes = user_args.get<uint32_t>(
      "mach_num_hashes", "integer",
      autotuneMachNumHashes(n_target_classes, output_range));

  _classifier = utils::Classifier::make(
      utils::buildModel(
          /* input_dim= */ tabular_options.feature_hash_range,
          /* output_dim= */ output_range,
          /* args= */ user_args, /* model_config= */ model_config,
          /* use_sigmoid_bce = */ true),
      user_args.get<bool>("freeze_hash_tables", "boolean",
                          defaults::FREEZE_HASH_TABLES));

  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart? is this why we're getting bad msmarco accuracy?

  // TODO(david) move things like label block and coldstart out of here and
  // into a classifier utils file?

  if (integer_target) {
    _mach_index = dataset::NumericCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes);
  } else {
    _mach_index = dataset::StringCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes,
        /* max_elements = */ n_target_classes);
  }

  auto mach_label_block = dataset::MachBlock::make(target_name, _mach_index,
                                                   target_config->delimiter);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{_multi_hash_label_block},
      std::set<std::string>{target_name}, tabular_options, force_parallel);
}

py::object UDTMachClassifier::train(
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

  return _classifier->train(
      train_dataset_loader, learning_rate, epochs, validation_dataset_loader,
      batch_size_opt, max_in_memory_batches, metrics, callbacks, verbose,
      logging_interval, licensing::TrainPermissionsToken(data));
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       bool verbose, bool return_metrics) {
  // TODO(david) should we throw an error if these flags are specified?
  (void)return_predicted_class;
  (void)return_metrics;

  // TODO(david) how should we process metrics? should we allow the user to pass
  // in metrics? maybe we limit it to only precision and recall @ K? maybe we
  // just ignore the user and give precision and recall @ K anyways? how should
  // we calculate precision and recall? should we decode the raw activations
  // then calculate precision and recall?

  // I think what we should do is
  // 1) evaluate with no metrics, get the activations
  // 2) write a decode single method that given some activations and some
  // parameters, like B and k, and using the _index we created, return the
  // predicted original ids

  auto dataset = _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto loaded_data =
      dataset->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);
  auto [test_data, test_labels] =
      utils::splitDataLabels(std::move(loaded_data));

  auto [output_metrics, output] =
      _classifier->model()->evaluate(test_data, test_labels, eval_config);

  std::vector<std::vector<std::variant<std::string, uint32_t>>>
      predicted_entities;
  for (uint32_t i = 0; i < output.numSamples(); i++) {
    BoltVector output_activations = output.getSampleAsNonOwningBoltVector(i);
    auto predictions = machSingleDecode(output_activations);
    predicted_entities.push_back(predictions);
  }

  // TODO(david) calculate precision and recall here based on the test_labels
  // and the predicted entities

  // TODO(david) return the predicted documents into numpy form
}

/**
 * Given the output activations to a mach model, decode using the mach index
 * back to the original documents. Documents may be strings or integers.
 * TODO(david) implement the more efficient version.
 */
std::vector<std::variant<std::string, uint32_t>>
UDTMachClassifier::machSingleDecode(const BoltVector& output) {
  uint32_t B = 25;
  uint32_t K = 5;

  auto top_B = output.findKLargestActivations(B);

  std::unordered_map<std::variant<std::string, uint32_t>, double>
      entity_to_scores;
  while (!top_B.empty()) {
    auto [activation, active_neuron] = top_B.top();
    auto entities = _mach_index->entitiesByHash(active_neuron);
    for (const auto& entity : entities) {
      if (!entity_to_scores.count(entity)) {
        entity_to_scores[entity] = activation;
      } else {
        entity_to_scores[entity] += activation;
      }
    }
    top_B.pop();
  }

  std::vector<std::pair<std::variant<std::string, uint32_t>, double>>
      entity_scores(entity_to_scores.begin(), entity_to_scores.end());

  std::sort(entity_scores.begin(), entity_scores.end(),
            [](auto& left, auto& right) { return left.second < right.second; });

  K = std::max<uint32_t>(K, entity_scores.size());

  std::vector<std::variant<std::string, uint32_t>> top_K_scores;
  std::transform(entity_scores.begin(), entity_scores.begin() + K,
                 std::back_inserter(top_K_scores),
                 [](const auto& pair) { return pair.first; });

  return top_K_scores;
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class) {
  BoltVector output = _classifier->model()->predictSingle(
      _dataset_factory->featurizeInput(sample), sparse_inference);
  auto decoded_output = machSingleDecode(output);
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class) {
  return _classifier->predictBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference,
      return_predicted_class);
}

py::object UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<ValidationDataSource>& validation,
    const std::vector<bolt::CallbackPtr>& callbacks, bool verbose) {
  auto data_source = utils::augmentColdStartData(
      data, strong_column_names, weak_column_names, _dataset_factory,
      /* integer_target = */ integerTarget(),
      /* label_column_name = */ _label_block->columnName(),
      /* label_delimiter = */ _label_block->delimiter());

  return train(data_source, learning_rate, epochs, validation,
               /* batch_size_opt = */ std::nullopt,
               /* max_in_memory_batches= */ std::nullopt, metrics,
               /* callbacks= */ callbacks, /* verbose= */ verbose,
               /* logging_interval= */ std::nullopt);
}

py::object UDTMachClassifier::embedding(const MapInput& sample) {}

py::object UDTMachClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {}

template <class Archive>
void UDTMachClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _classifier,
          _multi_hash_label_block, _dataset_factory);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)