#include "UDTMachClassifier.h"
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/model/Model.h>
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
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/backends/MachLogic.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/State.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachBlock.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/StringManipulation.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    config::ArgumentMap user_args) {
  (void)target_config;

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
  bolt::ModelPtr model = utils::buildModel(
      /* input_dim= */ input_dim, /* output_dim= */ num_buckets,
      /* args= */ user_args, /* model_config= */ model_config,
      /* use_sigmoid_bce = */ true, /* mach= */ true);
  bool freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                                defaults::FREEZE_HASH_TABLES);

  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart?

  if (!integer_target) {
    throw std::invalid_argument(
        "Option 'integer_target=True' must be specified when using extreme "
        "classification options.");
  }

  dataset::mach::MachIndexPtr mach_index = dataset::mach::MachIndex::make(
      /* num_buckets = */ num_buckets, /* num_hashes = */ num_hashes,
      /* num_elements = */ n_target_classes);

  _state = thirdai::data::State::make(mach_index);

  float mach_sampling_threshold = user_args.get<float>(
      "mach_sampling_threshold", "float", defaults::MACH_SAMPLING_THRESHOLD);

  bool rlhf = user_args.get<bool>("rlhf", "bool", false);
  size_t num_balancing_docs = user_args.get<uint32_t>(
      "rlhf_balancing_docs", "int", defaults::MAX_BALANCING_DOCS);
  size_t num_balancing_samples_per_doc =
      user_args.get<uint32_t>("rlhf_balancing_samples_per_doc", "int",
                              defaults::MAX_BALANCING_SAMPLES_PER_DOC);
  _logic =
      MachLogic(input_data_types, temporal_tracking_relationships, target_name,
                integer_target, model, freeze_hash_tables, num_buckets,
                mach_sampling_threshold, rlhf, num_balancing_docs,
                num_balancing_samples_per_doc, tabular_options);
}

py::object UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  return _logic.train(data, _state, learning_rate, epochs, train_metrics,
                      val_data, val_metrics, callbacks, options, comm);
}

py::object UDTMachClassifier::trainBatch(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  // TODO(Nicholas): Add back metrics
  (void)metrics;
  return _logic.trainBatch(batch, *_state, learning_rate);
}

py::object UDTMachClassifier::trainWithHashes(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  (void)metrics;
  return _logic.trainWithHashes(batch, *_state, learning_rate);
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose,
                                       std::optional<uint32_t> top_k) {
  (void)top_k;
  return _logic.evaluate(data, _state, metrics, sparse_inference, verbose);
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class,
                                      std::optional<uint32_t> top_k) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  return _logic.predict(sample, *_state, sparse_inference, top_k);
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class,
                                           std::optional<uint32_t> top_k) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  return _logic.predictBatch(samples, *_state, sparse_inference, top_k);
}

py::object UDTMachClassifier::predictBatch(
    const MapInputBatch& samples, bool sparse_inference,
    bool return_predicted_class, std::optional<uint32_t> top_k,
    const std::optional<std::vector<uint32_t>>& id_range) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  if (!id_range) {
    return _logic.predictBatch(samples, *_state, sparse_inference, top_k);
  }

  auto mach_index_subset = _state->machIndex()->subset(id_range.value());
  auto old_mach_index = _state->setMachIndex(mach_index_subset);
  auto results = _logic.predictBatch(samples, *_state, sparse_inference, top_k);
  _state->setMachIndex(old_mach_index);

  return results;
}

py::object UDTMachClassifier::predictHashes(
    const MapInput& sample, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  return _logic.predictHashes(sample, *_state, sparse_inference,
                              force_non_empty, num_hashes);
}

py::object UDTMachClassifier::predictHashesBatch(
    const MapInputBatch& samples, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  return _logic.predictHashesBatch(samples, *_state, sparse_inference,
                                   force_non_empty, num_hashes);
}

py::object UDTMachClassifier::outputCorrectness(
    const MapInputBatch& samples, const std::vector<uint32_t>& labels,
    bool sparse_inference, std::optional<uint32_t> num_hashes) {
  return _logic.outputCorrectness(samples, *_state, labels, sparse_inference,
                                  num_hashes);
}

void UDTMachClassifier::setModel(const ModelPtr& model) {
  _logic.setModel(model);
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
  return _logic.coldstart(data, _state, strong_column_names, weak_column_names,
                          learning_rate, epochs, train_metrics, val_data,
                          val_metrics, callbacks, options, comm);
}

py::object UDTMachClassifier::embedding(const MapInputBatch& sample) {
  return _logic.embedding(sample, *_state);
}

py::object UDTMachClassifier::entityEmbedding(const Label& label) {
  return _logic.entityEmbedding(label, *_state);
}

void UDTMachClassifier::introduceDocuments(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool fast_approximation, bool verbose) {
  (void)verbose;
  // TODO(Nicholas): add progress bar here.
  _logic.introduceDocuments(data, *_state, strong_column_names,
                            weak_column_names, num_buckets_to_sample_opt,
                            num_random_hashes, fast_approximation);
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes) {
  _logic.introduceDocument(document, *_state, strong_column_names,
                           weak_column_names, new_label, num_buckets_to_sample,
                           num_random_hashes);
}

struct BucketScore {
  uint32_t frequency = 0;
  float score = 0.0;
};

struct CompareBuckets {
  bool operator()(const std::pair<uint32_t, BucketScore>& lhs,
                  const std::pair<uint32_t, BucketScore>& rhs) {
    if (lhs.second.frequency == rhs.second.frequency) {
      return lhs.second.score > rhs.second.score;
    }
    return lhs.second.frequency > rhs.second.frequency;
  }
};

void UDTMachClassifier::introduceLabel(
    const MapInputBatch& samples, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  _logic.introduceLabel(samples, *_state, new_label, num_buckets_to_sample_opt,
                        num_random_hashes);
}

void UDTMachClassifier::forget(const Label& label) {
  MachLogic::forget(label, *_state);
}

void UDTMachClassifier::associate(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
  _logic.associate(rlhf_samples, *_state, n_buckets, n_association_samples,
                   n_balancing_samples, learning_rate, epochs);
}

void UDTMachClassifier::upvote(
    const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
    uint32_t n_upvote_samples, uint32_t n_balancing_samples,
    float learning_rate, uint32_t epochs) {
  _logic.upvote(rlhf_samples, *_state, n_upvote_samples, n_balancing_samples,
                learning_rate, epochs);
}

py::object UDTMachClassifier::associateTrain(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  return _logic.associateTrain(balancing_data, *_state, rlhf_samples, n_buckets,
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
  warnOnNonHashBasedMetrics(metrics);

  // TODO(nicholas): make sure max_in_memory_batches is none

  return _logic.associateColdStart(balancing_data, *_state, strong_column_names,
                                   weak_column_names, rlhf_samples, n_buckets,
                                   n_association_samples, learning_rate, epochs,
                                   metrics, options);
}

void UDTMachClassifier::setDecodeParams(uint32_t top_k_to_return,
                                        uint32_t num_buckets_to_eval) {
  _logic.setDecodeParams(top_k_to_return, *_state, num_buckets_to_eval);
}

void UDTMachClassifier::setIndex(const dataset::mach::MachIndexPtr& index) {
  // block allows indexes with different number of hashes but not output ranges
  _state->setMachIndex(index);
}

void UDTMachClassifier::setMachSamplingThreshold(float threshold) {
  _logic.setMachSamplingThreshold(threshold);
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

bolt::TensorPtr UDTMachClassifier::placeholderDocIds(uint32_t batch_size) {
  return bolt::Tensor::sparse(batch_size, std::numeric_limits<uint32_t>::max(),
                              /* nonzeros= */ 1);
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
  archive(cereal::base_class<UDTBackend>(this), _logic, _state);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)