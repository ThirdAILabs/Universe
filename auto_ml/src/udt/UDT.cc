#include "UDT.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/utils/Timer.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/pretrained/PretrainedBase.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/DeprecatedUDTMachClassifier.h>
#include <auto_ml/src/udt/backends/UDTClassifier.h>
#include <auto_ml/src/udt/backends/UDTGraphClassifier.h>
#include <auto_ml/src/udt/backends/UDTMach.h>
#include <auto_ml/src/udt/backends/UDTNer.h>
#include <auto_ml/src/udt/backends/UDTQueryReformulation.h>
#include <auto_ml/src/udt/backends/UDTRecurrentClassifier.h>
#include <auto_ml/src/udt/backends/UDTRegression.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/CheckLicense.h>
#include <pybind11/pytypes.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace thirdai::automl::udt {

struct BestScore {
  bool operator()(const std::pair<uint32_t, float>& a,
                  const std::pair<uint32_t, float>& b) {
    return a.second > b.second;
  }
};

UDT::UDT(ColumnDataTypes data_types,
         const UserProvidedTemporalRelationships& temporal_relationships,
         const std::string& target, char delimiter,
         const std::optional<std::string>& model_config,
         const py::object& pretrained_model,
         const config::ArgumentMap& user_args) {
  if (!data_types.count(target)) {
    throw std::invalid_argument("Target column '" + target +
                                "' not found in data types.");
  }

  if (!pretrained_model.is_none()) {
    if (auto categorical = asCategorical(data_types.at(target))) {
      _backend = std::make_unique<UDTClassifier>(
          data_types, categorical->expectNClasses(), categorical->isInteger(),
          pretrained_model.cast<PretrainedBasePtr>(), delimiter, user_args);
      return;
    }
    if (auto tags = asTokenTags(data_types.at(target))) {
      auto udt = pretrained_model.cast<std::shared_ptr<UDT>>();
      if (auto* udt_ner = dynamic_cast<UDTNer*>(udt->_backend.get())) {
        _backend = std::make_unique<UDTNer>(data_types, tags, target, udt_ner,
                                            user_args);
        return;
      }
      throw std::invalid_argument(
          "Only UDT NER models can be used as the base pretrained model for "
          "NER tasks.");
    }
    throw std::invalid_argument(
        "Pretrained models are only supported for classification or NER "
        "tasks.");
  }

  TabularOptions tabular_options;
  tabular_options.contextual_columns = user_args.get<bool>(
      "contextual_columns", "boolean", defaults::CONTEXTUAL_COLUMNS);
  tabular_options.time_granularity = user_args.get<std::string>(
      "time_granularity", "str", defaults::TIME_GRANULARITY);
  tabular_options.lookahead =
      user_args.get<uint32_t>("lookahead", "integer", defaults::LOOKAHEAD);
  tabular_options.delimiter = delimiter;
  tabular_options.feature_hash_range = user_args.get<uint32_t>(
      "input_dim", "integer", defaults::FEATURE_HASH_RANGE);
  if (user_args.contains("fhr")) {
    // For the QT app distribution we want to be able to override the input_dim
    // without revealing any information about the architecture.
    tabular_options.feature_hash_range =
        user_args.get<uint32_t>("fhr", "integer");
  }

  bool has_graph_inputs = hasGraphInputs(data_types);
  auto as_categorical = asCategorical(data_types.at(target));
  auto as_numerical = asNumerical(data_types.at(target));
  auto as_sequence = asSequence(data_types.at(target));
  auto as_tags = asTokenTags(data_types.at(target));
  auto as_text = asText(data_types.at(target));

  if (as_categorical && has_graph_inputs) {
    // TODO(Any): Add support for model config and user args
    _backend = std::make_unique<UDTGraphClassifier>(data_types, as_categorical,
                                                    target, tabular_options);
  } else if (as_categorical && !has_graph_inputs) {
    bool use_mach =
        user_args.get<bool>("extreme_classification", "boolean",
                            defaults::USE_MACH) ||
        user_args.get<bool>("neural_db", "boolean", defaults::USE_MACH);
    if (use_mach) {
      if (user_args.get<bool>("v1", "boolean", false)) {
        _backend = std::make_unique<UDTMachClassifier>(
            data_types, temporal_relationships, target, as_categorical,
            as_categorical->expectNClasses(), as_categorical->isInteger(),
            tabular_options, model_config, user_args);
      } else {
        _backend = std::make_unique<UDTMach>(
            data_types, temporal_relationships, target, as_categorical,
            tabular_options, model_config, user_args);
      }
    } else {
      _backend = std::make_unique<UDTClassifier>(
          data_types, temporal_relationships, target, as_categorical,
          tabular_options, model_config, user_args);
    }
  } else if (as_numerical && !has_graph_inputs) {
    _backend = std::make_unique<UDTRegression>(
        data_types, temporal_relationships, target, as_numerical,
        as_numerical->explicit_granularity, tabular_options, model_config,
        user_args);
  } else if (as_sequence && !has_graph_inputs) {
    _backend = std::make_unique<UDTRecurrentClassifier>(
        data_types, temporal_relationships, target, as_sequence,
        tabular_options, model_config, user_args);
  } else if (as_tags) {
    _backend = std::make_unique<UDTNer>(data_types, as_tags, target, nullptr,
                                        user_args);
  } else if (as_text) {
    _backend = std::make_unique<UDTQueryReformulation>(
        data_types, target, delimiter, model_config, user_args);
  } else {
    throwUnsupportedUDTConfigurationError(data_types.at(target),
                                          has_graph_inputs);
  }
}

py::object UDT::train(const dataset::DataSourcePtr& data, float learning_rate,
                      uint32_t epochs,
                      const std::vector<std::string>& train_metrics,
                      const dataset::DataSourcePtr& val_data,
                      const std::vector<std::string>& val_metrics,
                      const std::vector<CallbackPtr>& callbacks,
                      TrainOptions options,
                      const bolt::DistributedCommPtr& comm, py::kwargs kwargs) {
  licensing::entitlements().verifyDataSource(data);

  auto output =
      _backend->train(data, learning_rate, epochs, train_metrics, val_data,
                      val_metrics, callbacks, options, comm, std::move(kwargs));

  return output;
}

py::object UDT::trainBatch(const MapInputBatch& batch, float learning_rate) {
  licensing::entitlements().verifyFullAccess();

  auto output = _backend->trainBatch(batch, learning_rate);

  return output;
}

void UDT::setOutputSparsity(float sparsity, bool rebuild_hash_tables) {
  _backend->setOutputSparsity(sparsity, rebuild_hash_tables);
}

py::object UDT::evaluate(const dataset::DataSourcePtr& data,
                         const std::vector<std::string>& metrics,
                         bool sparse_inference, bool verbose,
                         py::kwargs kwargs) {
  auto result = _backend->evaluate(data, metrics, sparse_inference, verbose,
                                   std::move(kwargs));

  return result;
}

py::object UDT::predict(const MapInput& sample, bool sparse_inference,
                        bool return_predicted_class,
                        std::optional<uint32_t> top_k,
                        const py::kwargs& kwargs) {
  auto result = _backend->predict(sample, sparse_inference,
                                  return_predicted_class, top_k, kwargs);

  return result;
}

py::object UDT::predictBatch(const MapInputBatch& sample, bool sparse_inference,
                             bool return_predicted_class,
                             std::optional<uint32_t> top_k,
                             const py::kwargs& kwargs) {
  auto result = _backend->predictBatch(sample, sparse_inference,
                                       return_predicted_class, top_k, kwargs);

  return result;
}

py::object UDT::scoreBatch(const MapInputBatch& samples,
                           const std::vector<std::vector<Label>>& classes,
                           std::optional<uint32_t> top_k) {
  auto result = _backend->scoreBatch(samples, classes, top_k);

  return result;
}

std::vector<std::pair<std::string, float>> UDT::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  auto result = _backend->explain(sample, target_class);

  return result;
}

py::object UDT::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm, const py::kwargs& kwargs) {
  licensing::entitlements().verifyDataSource(data);

  return _backend->coldstart(data, strong_column_names, weak_column_names,
                             variable_length, learning_rate, epochs,
                             train_metrics, val_data, val_metrics, callbacks,
                             options, comm, kwargs);
}

std::vector<uint32_t> UDT::modelDims() const {
  std::vector<uint32_t> dims;
  for (const auto& comp : model()->computationOrder()) {
    dims.push_back(comp->dim());
  }

  return dims;
}

void UDT::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void UDT::save_stream(std::ostream& output) const {
  const_cast<UDT*>(this)->_save_optimizer = false;
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(*this);
}

void UDT::checkpoint(const std::string& filename) const {
  std::ofstream output =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);

  const_cast<UDT*>(this)->_save_optimizer = true;
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(*this);
}

std::shared_ptr<UDT> UDT::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<UDT> UDT::load_stream(std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);
  std::shared_ptr<UDT> deserialize_into(new UDT());
  iarchive(*deserialize_into);

  return deserialize_into;
}

bool UDT::hasGraphInputs(const ColumnDataTypes& data_types) {
  uint64_t neighbor_col_count = 0;
  uint64_t node_id_col_count = 0;
  for (const auto& [col_name, data_type] : data_types) {
    if (asNeighbors(data_type)) {
      neighbor_col_count++;
    }
    if (asNodeID(data_type)) {
      node_id_col_count++;
    }
  }
  if (neighbor_col_count == 0 && node_id_col_count == 0) {
    return false;
  }
  if (neighbor_col_count == 1 && node_id_col_count == 1) {
    return true;
  }
  throw std::invalid_argument(
      "Expected either 1 of both neighbor and node id data types (for a graph "
      "learning problem) or 0 of both (for a non-graph learning problem). "
      "Instead, found " +
      std::to_string(neighbor_col_count) + " neighbor data types and " +
      std::to_string(node_id_col_count) +
      " node id data types.\nRefer to "
      "https://github.com/ThirdAILabs/Demos/blob/main/"
      "universal_deep_transformer/graph_neural_networks/"
      "GraphNodeClassification.ipynb for more details on how to use a "
      "UniversalDeepTransformer for a Graph Classification Problem.");
}

std::unique_ptr<UDTBackend> backendFromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

  if (type == UDTClassifier::type()) {
    return UDTClassifier::fromArchive(archive);
  }
  if (type == UDTGraphClassifier::type()) {
    return UDTGraphClassifier::fromArchive(archive);
  }
  if (type == UDTMach::type()) {
    return UDTMach::fromArchive(archive);
  }
  if (type == UDTQueryReformulation::type()) {
    return UDTQueryReformulation::fromArchive(archive);
  }
  if (type == UDTRecurrentClassifier::type()) {
    return UDTRecurrentClassifier::fromArchive(archive);
  }
  if (type == UDTRegression::type()) {
    return UDTRegression::fromArchive(archive);
  }
  if (type == UDTNer::type()) {
    return UDTNer::fromArchive(archive);
  }
  throw std::invalid_argument("Invalid backend type '" + type + "'.");
}

template <class Archive>
void UDT::save(Archive& archive, const uint32_t version) const {
  (void)version;
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);

  auto thirdai_archive = _backend->toArchive(_save_optimizer);

  archive(thirdai_archive);
}

template <class Archive>
void UDT::load(Archive& archive, const uint32_t version) {
  std::string thirdai_version;
  archive(thirdai_version);

  if (version <= versions::UDT_LAST_OLD_SERIALIZATION_VERSION) {
    std::string class_name = "UDT_BASE";
    // We use the UDT_LAST_OLD_SERIALIZATION_VERSION as the current version
    // becuase that's the version that we're using to load the model using the
    // old cereal code.
    versions::checkVersion(version,
                           versions::UDT_LAST_OLD_SERIALIZATION_VERSION,
                           thirdai_version, thirdai::version(), class_name);

    archive(_backend);

    migrateToMachV2();
  } else {
    ar::ArchivePtr thirdai_archive;
    archive(thirdai_archive);

    _backend = backendFromArchive(*thirdai_archive);
  }
}

void UDT::migrateToMachV2() {
  if (auto* old_mach = dynamic_cast<UDTMachClassifier*>(_backend.get())) {
    _backend = std::make_unique<UDTMach>(old_mach->getMachInfo());
  }
}

void UDT::throwUnsupportedUDTConfigurationError(const DataTypePtr& target,
                                                bool has_graph_inputs) {
  std::stringstream error_msg;
  error_msg << "Target data type " << target->typeName() << " is not valid";

  if (has_graph_inputs) {
    error_msg
        << " for a UniversalDeepTransformer model with graph classification";
  }
  error_msg << ".\nThe following target types are supported to initialize a "
               "UniversalDeepTransformer: "
            << std::endl
            << std::endl;
  error_msg << "* Graph Classification -> bolt.types.categorical(type='int', "
               "n_classes=<num_classes>)"
            << std::endl;
  error_msg << "* Text or Tabular Classification -> "
               "bolt.types.categorical(type='int' or 'str', "
               "n_classes=<num_classes>)"
            << std::endl;
  error_msg << "* Extreme Classification -> bolt.types.categorical(type='int', "
               "n_classes=<num_classes>)"
            << std::endl;
  error_msg << "* Regression -> "
               "bolt.types.numerical(range=(<expected_lower_limit>, "
               "<expected_upper_limit>))"
            << std::endl;
  error_msg << "* RecurrentClassifier -> "
               "bolt.types.sequence(n_classes=<num_classes>, "
               "max_length=<maximum_labels_in_sequence>)"
            << std::endl;
  error_msg << "* Named Entity Recognition -> "
               "bolt.types.token_tags(tags=<list_of_tag_strings>, "
               "default_tag=<ordinary_entity_tag>)"
            << std::endl;
  error_msg << "* Query Reformulation -> bolt.types.text()" << std::endl
            << std::endl
            << std::endl;

  error_msg << "Refer to https://thirdailabs.github.io/thirdaibolt.html for "
               "more details on how to use a UniversalDeepTransformer model "
               "for a specific task."
            << std::endl;

  throw std::invalid_argument(error_msg.str());
}

bool UDT::isV1() const {
  return dynamic_cast<UDTMachClassifier*>(_backend.get());
}

std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>>
UDT::parallelInference(const std::vector<std::shared_ptr<UDT>>& models,
                       const MapInputBatch& batch, bool sparse_inference,
                       std::optional<uint32_t> top_k) {
  std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>> outputs(
      models.size());

  bool non_mach = false;
#pragma omp parallel for default(none)                      \
    shared(models, batch, outputs, sparse_inference, top_k, \
           non_mach) if (batch.size() == 1)
  for (size_t i = 0; i < models.size(); i++) {
    if (auto* mach = dynamic_cast<UDTMach*>(models[i]->_backend.get())) {
      outputs[i] = mach->predictBatchImpl(
          batch, sparse_inference, /*return_predicted_class*/ false, top_k);
    } else if (auto* mach = dynamic_cast<UDTMachClassifier*>(
                   models[i]->_backend.get())) {
      outputs[i] = mach->predictImpl(batch, sparse_inference, top_k);
    } else {
#pragma omp critical
      non_mach = true;
    }
  }

  if (non_mach) {
    throw std::invalid_argument(
        "Cannot perform parallel inference on non mach model.");
  }

  return outputs;
}

std::vector<std::vector<UDT::Scores>> UDT::labelProbeMultipleShards(
    const std::vector<std::vector<std::shared_ptr<UDT>>>& shards,
    const MapInputBatch& batch, bool sparse_inference,
    std::optional<uint32_t> top_k) {
  std::vector<std::vector<UDT::Scores>> shard_scores(shards.size());

#pragma omp parallel for default(none)                    \
    shared(shard_scores, shards, batch, sparse_inference, \
           top_k) if (batch.size() == 1)
  for (size_t shard_id = 0; shard_id < shards.size(); shard_id++) {
    shard_scores[shard_id] = labelProbeMultipleMach(shards[shard_id], batch,
                                                    sparse_inference, top_k);
  }
  return shard_scores;
}

std::vector<UDT::Scores> UDT::labelProbeMultipleMach(
    const std::vector<std::shared_ptr<UDT>>& models, const MapInputBatch& batch,
    bool sparse_inference, std::optional<uint32_t> top_k) {
  if (models.empty()) {
    throw std::invalid_argument(
        "Atleast 1 model should be passed for decoding");
  }

  std::vector<UDTMach*> mach_models;
  for (const auto& model : models) {
    if (auto* mach_model = dynamic_cast<UDTMach*>(model->_backend.get())) {
      auto num_hashes = mach_model->getIndex()->numHashes();
      if (num_hashes > 1) {
        throw std::invalid_argument(
            "Cannot perform label probing for a mach model with more than 1 "
            "hash");
      }
      mach_models.push_back(mach_model);
    } else {
      throw std::invalid_argument("Cannot perform decoding on non mach model.");
    }
  }

  bolt::TensorList scores(mach_models.size());

#pragma omp parallel for default(none) shared( \
    mach_models, scores, batch, sparse_inference) if (batch.size() == 1)
  for (size_t i = 0; i < mach_models.size(); i++) {
    auto output = mach_models[i]->model()->forward(
        mach_models[i]->featurizer()->featurizeInputBatch(batch),
        sparse_inference);
    scores[i] = output.at(0);
  }

  auto top_k_to_return = top_k.value_or(mach_models[0]->defaultTopKToReturn());

  std::vector<Scores> output(batch.size());

  // TODO(Shubh): Add support for lossy decoding to make inference faster.
#pragma omp parallel for default(none) shared( \
    batch, scores, top_k_to_return, output, mach_models) if (batch.size() > 1)
  for (size_t i = 0; i < batch.size(); i++) {
    std::vector<std::unordered_set<uint32_t>> individual_candidates(
        mach_models.size());
    std::exception_ptr error;
#pragma omp parallel for default(none)                    \
    shared(mach_models, individual_candidates, scores, i, \
           error) if (batch.size() == 1)
    for (size_t m = 0; m < mach_models.size(); m++) {
      try {
        const auto& index = mach_models[m]->getIndex();
        auto top_model_candidates = scores[m]->getVector(i).topKNeurons(
            mach_models[m]->numBucketsToEval());

        while (!top_model_candidates.empty()) {
          uint32_t bucket = top_model_candidates.top().second;
          for (uint32_t id : index->getEntities(bucket)) {
            individual_candidates[m].insert(id);
          }
          top_model_candidates.pop();
        }
      } catch (...) {
#pragma omp critical
        error = std::current_exception();
      }
    }

    if (error) {
      std::rethrow_exception(error);
    }

    std::unordered_set<uint32_t> global_candidates;
    for (const auto& candidate_set : individual_candidates) {
      global_candidates.insert(candidate_set.begin(), candidate_set.end());
    }

    std::vector<std::pair<uint32_t, float>> global_candidate_scores;
    global_candidate_scores.reserve(global_candidates.size());
    for (uint32_t candidate : global_candidates) {
      global_candidate_scores.emplace_back(candidate, 0);
    }

    std::vector<std::vector<std::pair<uint32_t, float>>>
        individual_candidate_scores(mach_models.size());

#pragma omp parallel for default(none)                                        \
    shared(mach_models, individual_candidate_scores, global_candidate_scores, \
           scores, i, error) if (batch.size() == 1)
    for (size_t m = 0; m < mach_models.size(); m++) {
      individual_candidate_scores[m] = global_candidate_scores;

      try {
        const auto& index = mach_models[m]->getIndex();
        const BoltVector& score_vec = scores[m]->getVector(i);
        if (score_vec.isDense()) {
          const float* activations = scores[m]->getVector(i).activations;
          for (auto& [id, score] : individual_candidate_scores[m]) {
            score += activations[index->getHashes(id)[0]];
          }
        } else {
          std::unordered_map<uint32_t, float> score_map;
          for (size_t j = 0; j < score_vec.len; j++) {
            score_map[score_vec.active_neurons[j]] = score_vec.activations[j];
          }
          for (auto& [id, score] : individual_candidate_scores[m]) {
            uint32_t hash = index->getHashes(id)[0];
            if (score_map.count(hash)) {
              score += score_map[hash];
            }
          }
        }
      } catch (...) {
#pragma omp critical
        error = std::current_exception();
      }
    }

    for (const auto& scores : individual_candidate_scores) {
      for (size_t i = 0; i < global_candidate_scores.size(); i++) {
        global_candidate_scores[i].second += scores[i].second;
      }
    }

    if (error) {
      std::rethrow_exception(error);
    }

    std::sort(global_candidate_scores.begin(), global_candidate_scores.end(),
              BestScore{});
    if (global_candidate_scores.size() > top_k_to_return) {
      global_candidate_scores.resize(top_k_to_return);
    }

    output[i] = std::move(global_candidate_scores);
  }

  return output;
}

size_t UDT::estimateHashTableSize(size_t output_dim,
                                  std::optional<float> sparsity) {
  return bolt::DWTASamplingConfig::estimateHashTableSize(
      output_dim, sparsity.value_or(utils::autotuneSparsity(output_dim)));
}

}  // namespace thirdai::automl::udt

CEREAL_CLASS_VERSION(thirdai::automl::udt::UDT,
                     thirdai::versions::UDT_BASE_VERSION)