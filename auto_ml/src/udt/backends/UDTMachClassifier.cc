#include "UDTMachClassifier.h"
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/MachPrecision.h>
#include <bolt/src/train/metrics/MachRecall.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/MachFeaturizer.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/ColdStartText.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachBlock.h>
#include <mach/src/Mach.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/StringManipulation.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::udt {

using bolt::metrics::LossMetric;
using bolt::metrics::MachPrecision;
using bolt::metrics::MachRecall;
using bolt::metrics::PrecisionAtK;
using bolt::metrics::RecallAtK;

uint32_t expectInteger(const Label& label) {
  if (!std::holds_alternative<uint32_t>(label)) {
    throw std::invalid_argument("Must use integer label.");
  }
  return std::get<uint32_t>(label);
}

UDTMachClassifier::UDTMachClassifier(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target,
    const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    config::ArgumentMap user_args)
    : _default_top_k_to_return(defaults::MACH_TOP_K_TO_RETURN),
      _num_buckets_to_eval(defaults::MACH_NUM_BUCKETS_TO_EVAL) {
  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart?

  if (!integer_target) {
    throw std::invalid_argument(
        "Option 'integer_target=True' must be specified when using extreme "
        "classification options.");
  }

  auto temporal_relationships = TemporalRelationshipsAutotuner::autotune(
      input_data_types, temporal_tracking_relationships,
      tabular_options.lookahead);

  _data = MachFeaturizer::make(input_data_types, target_config,
                               temporal_relationships, target_name,
                               tabular_options);

  uint32_t input_dim = tabular_options.feature_hash_range;

  if (user_args.get<bool>("neural_db", "boolean", /* default_val= */ false)) {
    input_dim = 50000;
    user_args.insert<uint32_t>("embedding_dimension", 2048);
    user_args.insert<uint32_t>("extreme_output_dim", 50000);
    user_args.insert<uint32_t>("extreme_num_hashes", 8);
    user_args.insert<bool>("use_bias", false);
    user_args.insert<bool>("use_tanh", true);
    user_args.insert<bool>("rlhf", true);
  }

  uint32_t num_buckets = user_args.get<uint32_t>(
      "extreme_output_dim", "integer", autotuneMachOutputDim(n_target_classes));
  uint32_t num_hashes = user_args.get<uint32_t>(
      "extreme_num_hashes", "integer",
      autotuneMachNumHashes(n_target_classes, num_buckets));
  float mach_sampling_threshold = user_args.get<float>(
      "mach_sampling_threshold", "float", defaults::MACH_SAMPLING_THRESHOLD);
  bool freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                                defaults::FREEZE_HASH_TABLES);

  _classifier = mach::Mach::make(
      input_dim, num_buckets, user_args, model_config,
      /* use_sigmoid_bce = */ true, num_hashes, mach_sampling_threshold,
      freeze_hash_tables, _data->modelInputIndicesColumn(),
      _data->modelInputValuesColumn(), _data->modelLabelColumn(),
      _data->modelBucketColumn());

  if (user_args.get<bool>("rlhf", "bool", false)) {
    size_t num_balancing_docs = user_args.get<uint32_t>(
        "rlhf_balancing_docs", "int", defaults::MAX_BALANCING_DOCS);
    size_t num_balancing_samples_per_doc =
        user_args.get<uint32_t>("rlhf_balancing_samples_per_doc", "int",
                                defaults::MAX_BALANCING_SAMPLES_PER_DOC);

    _classifier->enableRlhf(num_balancing_docs, num_balancing_samples_per_doc);
  }

  std::vector<uint32_t> labels(n_target_classes);
  std::iota(labels.begin(), labels.end(), 0);
  _classifier->randomlyIntroduceEntities(data::ColumnMap(
      {{_data->modelLabelColumn(),
        data::ValueColumn<uint32_t>::make(
            std::move(labels),
            /* dim= */ std::numeric_limits<uint32_t>::max())}}));
}

py::object UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  return py::cast(_classifier->train(
      _data->labeledTransform(_data->iter(data)),
      val_data ? _data->labeledTransform(_data->iter(val_data)) : nullptr,
      learning_rate, epochs, getMetrics(train_metrics, "train_"),
      getMetrics(val_metrics, "val_"), callbacks, options, comm));
}

py::object UDTMachClassifier::trainBatch(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  _classifier->train(
      _data->labeledTransform(data::ColumnMap::fromMapInputBatch(batch)),
      learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::trainWithHashes(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  _classifier->trainBuckets(
      _data->bucketedTransform(data::ColumnMap::fromMapInputBatch(batch)),
      learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose,
                                       std::optional<uint32_t> top_k) {
  (void)top_k;
  return py::cast(_classifier->evaluate(
      _data->labeledTransform(_data->iter(data)), getMetrics(metrics, "eval_"),
      sparse_inference, verbose));
}

std::vector<std::vector<std::pair<uint32_t, double>>>
UDTMachClassifier::predictImpl(const MapInputBatch& samples,
                               bool sparse_inference,
                               bool return_predicted_class,
                               std::optional<uint32_t> top_k) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }
  return _classifier->predict(_data->constUnlabeledTransform(
                                  data::ColumnMap::fromMapInputBatch(samples)),
                              sparse_inference,
                              top_k.value_or(_default_top_k_to_return),
                              _num_buckets_to_eval);
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class,
                                      std::optional<uint32_t> top_k) {
  return py::cast(
      predictImpl({sample}, sparse_inference, return_predicted_class, top_k)
          .at(0));
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class,
                                           std::optional<uint32_t> top_k) {
  return py::cast(
      predictImpl(samples, sparse_inference, return_predicted_class, top_k));
}

py::object UDTMachClassifier::scoreBatch(
    const MapInputBatch& samples,
    const std::vector<std::vector<Label>>& classes,
    std::optional<uint32_t> top_k) {
  auto columns = data::ColumnMap::fromMapInputBatch(samples);
  columns = _data->constUnlabeledTransform(std::move(columns));
  std::vector<std::unordered_set<uint32_t>> entities(classes.size());
  for (uint32_t row = 0; row < classes.size(); row++) {
    entities[row].reserve(classes[row].size());
    for (const auto& entity : classes[row]) {
      entities[row].insert(expectInteger(entity));
    }
  }

  return py::cast(_classifier->score(columns, entities, top_k));
}

py::object UDTMachClassifier::predictHashes(
    const MapInput& sample, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  return py::cast(
      _classifier
          ->predictBuckets(_data->constUnlabeledTransform(
                               data::ColumnMap::fromMapInput(sample)),
                           sparse_inference, num_hashes, force_non_empty)
          .at(0));
}

py::object UDTMachClassifier::predictHashesBatch(
    const MapInputBatch& samples, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  return py::cast(_classifier->predictBuckets(
      _data->constUnlabeledTransform(
          data::ColumnMap::fromMapInputBatch(samples)),
      sparse_inference, num_hashes, force_non_empty));
}

py::object UDTMachClassifier::outputCorrectness(
    const MapInputBatch& samples, const std::vector<uint32_t>& labels,
    bool sparse_inference, std::optional<uint32_t> num_hashes) {
  return py::cast(_classifier->outputCorrectness(
      _data->constUnlabeledTransform(
          data::ColumnMap::fromMapInputBatch(samples)),
      labels, num_hashes, sparse_inference));
}

void UDTMachClassifier::setModel(const ModelPtr& model) {
  bolt::ModelPtr& curr_model = _classifier->model();

  utils::verifyCanSetModel(curr_model, model);

  curr_model = model;
}

py::object UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  auto train_iter = _data->labeledTransform(_data->coldstart(
      _data->iter(data), strong_column_names, weak_column_names));
  auto val_iter =
      val_data ? _data->labeledTransform(_data->iter(val_data)) : nullptr;

  return py::cast(_classifier->train(
      train_iter, val_iter, learning_rate, epochs,
      getMetrics(train_metrics, "train_"), getMetrics(val_metrics, "val_"),
      callbacks, options, comm));
}

py::object UDTMachClassifier::embedding(const MapInputBatch& sample) {
  return py::cast(_classifier->embedding(_data->constUnlabeledTransform(
      data::ColumnMap::fromMapInputBatch(sample))));
}

py::object UDTMachClassifier::entityEmbedding(const Label& label) {
  auto embedding = _classifier->entityEmbedding(expectInteger(label));
  NumpyArray<float> np_weights(embedding.size());
  std::copy(embedding.begin(), embedding.end(), np_weights.mutable_data());
  return std::move(np_weights);
}

void UDTMachClassifier::introduceDocuments(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool fast_approximation, bool verbose) {
  (void)verbose;
  auto columns = _data->coldstart(_data->columns(data), strong_column_names,
                                  weak_column_names, fast_approximation);

  _classifier->introduceEntities(_data->labeledTransform(std::move(columns)),
                                 num_buckets_to_sample_opt, num_random_hashes);
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes) {
  auto columns = _data->addLabelColumn(data::ColumnMap::fromMapInput(document),
                                       expectInteger(new_label));
  columns = _data->coldstart(std::move(columns), strong_column_names,
                             weak_column_names);
  columns = _data->labeledTransform(std::move(columns));
  _classifier->introduceEntities(columns, num_buckets_to_sample,
                                 num_random_hashes);
}

void UDTMachClassifier::introduceLabel(
    const MapInputBatch& samples, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  auto columns = _data->addLabelColumn(
      data::ColumnMap::fromMapInputBatch(samples), expectInteger(new_label));
  columns = _data->labeledTransform(std::move(columns));
  _classifier->introduceEntities(columns, num_buckets_to_sample_opt,
                                 num_random_hashes);
}

void UDTMachClassifier::forget(const Label& label) {
  _classifier->eraseEntity(expectInteger(label));
}

void UDTMachClassifier::associate(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
  auto [from_columns, to_columns] = _data->associationColumnMaps(rlhf_samples);
  from_columns = _data->constUnlabeledTransform(std::move(from_columns));
  to_columns = _data->constUnlabeledTransform(std::move(to_columns));
  _classifier->associate(from_columns, to_columns, learning_rate,
                         n_association_samples, n_balancing_samples, n_buckets,
                         epochs, defaults::ASSOCIATE_BATCH_SIZE);
}

void UDTMachClassifier::upvote(
    const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
    uint32_t n_upvote_samples, uint32_t n_balancing_samples,
    float learning_rate, uint32_t epochs) {
  auto upvote_columns = _data->upvoteLabeledColumnMap(rlhf_samples);
  // Apply unlabeled transform since `upvote_columns` already has a properly
  // formatted label column.
  upvote_columns = _data->constUnlabeledTransform(std::move(upvote_columns));
  _classifier->upvote(upvote_columns, learning_rate, n_upvote_samples,
                      n_balancing_samples, epochs,
                      defaults::ASSOCIATE_BATCH_SIZE);
}

py::object UDTMachClassifier::associateTrain(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto train_columns = _data->columns(balancing_data);
  return associateTrainImpl(std::move(train_columns), rlhf_samples, n_buckets,
                            n_association_samples, learning_rate, epochs,
                            metrics, options);
}

py::object UDTMachClassifier::associateColdStart(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto train_columns = _data->columns(balancing_data);
  train_columns = _data->coldstart(std::move(train_columns),
                                   strong_column_names, weak_column_names);
  return associateTrainImpl(std::move(train_columns), rlhf_samples, n_buckets,
                            n_association_samples, learning_rate, epochs,
                            metrics, options);
}

py::object UDTMachClassifier::associateTrainImpl(
    data::ColumnMap&& train_columns,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  warnOnNonHashBasedMetrics(metrics);
  auto [from_columns, to_columns] = _data->associationColumnMaps(rlhf_samples);
  from_columns = _data->constUnlabeledTransform(std::move(from_columns));
  to_columns = _data->constUnlabeledTransform(std::move(to_columns));
  train_columns = _data->labeledTransform(std::move(train_columns));
  return py::cast(_classifier->associateTrain(
      std::move(from_columns), to_columns, std::move(train_columns),
      learning_rate, n_association_samples, n_buckets, epochs,
      options.batchSize(), getMetrics(metrics, "train_"), options));
}

void UDTMachClassifier::setDecodeParams(uint32_t top_k_to_return,
                                        uint32_t num_buckets_to_eval) {
  if (top_k_to_return == 0 || num_buckets_to_eval == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  uint32_t num_buckets = getIndex()->numBuckets();
  if (num_buckets_to_eval > num_buckets) {
    throw std::invalid_argument(
        "Cannot eval with num_buckets_to_eval greater than " +
        std::to_string(num_buckets) + ".");
  }

  uint32_t num_classes = getIndex()->numEntities();
  if (top_k_to_return > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to "
        "predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  _default_top_k_to_return = top_k_to_return;
  _num_buckets_to_eval = num_buckets_to_eval;
}

void UDTMachClassifier::setMachSamplingThreshold(float threshold) {
  _classifier->setMachSamplingThreshold(threshold);
}

InputMetrics UDTMachClassifier::getMetrics(
    const std::vector<std::string>& metric_names, const std::string& prefix) {
  const auto& model = _classifier->model();
  if (model->outputs().size() != 1 || model->labels().size() != 2 ||
      model->losses().size() != 1) {
    throw std::invalid_argument(
        "Expected model to have single input, two labels, and one "
        "loss.");
  }

  bolt::ComputationPtr output = model->outputs().front();
  bolt::ComputationPtr hash_labels = model->labels().front();
  bolt::ComputationPtr true_class_labels = model->labels().back();
  bolt::LossPtr loss = model->losses().front();

  InputMetrics metrics;
  for (const auto& name : metric_names) {
    if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 10, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachPrecision>(
          getIndex(), _num_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachRecall>(
          getIndex(), _num_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("hash_precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 15, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<PrecisionAtK>(output, hash_labels, k);
    } else if (std::regex_match(name, std::regex("hash_recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 12, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<RecallAtK>(output, hash_labels, k);
    } else if (name == "loss") {
      metrics[prefix + name] = std::make_shared<LossMetric>(loss);
    } else {
      throw std::invalid_argument(
          "Invalid metric '" + name +
          "'. Please use precision@k, recall@k, or loss.");
    }
  }

  return metrics;
}

void UDTMachClassifier::warnOnNonHashBasedMetrics(
    const std::vector<std::string>& metrics) {
  for (const auto& metric : metrics) {
    if (!std::regex_match(metric, std::regex("((hash_)|(loss)).*"))) {
      std::cerr << "Warning: using precision/recall with associate_train can "
                   "cause skewed results since the association samples may not "
                   "have a true label."
                << std::endl;
    }
  }
}

template void UDTMachClassifier::serialize(cereal::BinaryInputArchive&,
                                           const uint32_t version);
template void UDTMachClassifier::serialize(cereal::BinaryOutputArchive&,
                                           const uint32_t version);

template <class Archive>
void UDTMachClassifier::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_MACH_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_MACH_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_MACH_CLASSIFIER_VERSION after
  // serialization changes
  archive(cereal::base_class<UDTBackend>(this), _classifier, _data,
          _default_top_k_to_return, _num_buckets_to_eval);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)