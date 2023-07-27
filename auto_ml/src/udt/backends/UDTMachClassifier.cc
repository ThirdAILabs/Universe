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
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/ColumnMap.h>
#include <data/src/Loader.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/ColdStartText.h>
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

using bolt::train::metrics::LossMetric;
using bolt::train::metrics::MachPrecision;
using bolt::train::metrics::MachRecall;
using bolt::train::metrics::PrecisionAtK;
using bolt::train::metrics::RecallAtK;

UDTMachClassifier::UDTMachClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    uint32_t n_target_classes, bool integer_target,
    const data::TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    config::ArgumentMap user_args)
    : _default_top_k_to_return(defaults::MACH_TOP_K_TO_RETURN),
      _num_buckets_to_eval(defaults::MACH_NUM_BUCKETS_TO_EVAL) {
  if (!temporal_tracking_relationships.empty()) {
    throw std::invalid_argument(
        "Temporal tracking is not supported for this model type.");
  }

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

  _classifier = utils::Classifier::make(
      utils::buildModel(
          /* input_dim= */ input_dim, /* output_dim= */ num_buckets,
          /* args= */ user_args, /* model_config= */ model_config,
          /* use_sigmoid_bce = */ true, /* mach= */ true),
      user_args.get<bool>("freeze_hash_tables", "boolean",
                          defaults::FREEZE_HASH_TABLES));

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

  _data_factory =
      MachDatasetFactory(input_data_types, mach_index, tabular_options);

  _mach_sampling_threshold = user_args.get<float>(
      "mach_sampling_threshold", "float", defaults::MACH_SAMPLING_THRESHOLD);

  // TODO(David): Should we call this in constructor as well?
  updateSamplingStrategy();

  if (user_args.get<bool>("rlhf", "bool", false)) {
    size_t num_balancing_docs = user_args.get<uint32_t>(
        "rlhf_balancing_docs", "int", defaults::MAX_BALANCING_DOCS);
    size_t num_balancing_samples_per_doc =
        user_args.get<uint32_t>("rlhf_balancing_samples_per_doc", "int",
                                defaults::MAX_BALANCING_SAMPLES_PER_DOC);

    _rlhf_sampler = std::make_optional<RLHFSampler>(
        num_balancing_docs, num_balancing_samples_per_doc);
  }
}

py::object UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::train::DistributedCommPtr& comm) {
  thirdai::data::LoaderPtr val_dataset_loader;
  if (val_data) {
    val_dataset_loader =
        _data_factory.getDataLoader(val_data, /* include_mach_labels= */ true,
                                    options.shuffle_config, options.verbose);
  }

  addBalancingSamples(data);

  auto train_dataset_loader =
      _data_factory.getDataLoader(data, /* include_mach_labels= */ true,
                                  options.shuffle_config, options.verbose);

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            getMetrics(train_metrics, "train_"),
                            val_dataset_loader, getMetrics(val_metrics, "val_"),
                            callbacks, options, comm);
}

py::object UDTMachClassifier::trainBatch(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] =
      _data_factory.featurizeTrainingBatch(batch, /* prehashed= */ false);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::trainWithHashes(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] =
      _data_factory.featurizeTrainingBatch(batch, /* prehashed= */ true);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose,
                                       std::optional<uint32_t> top_k) {
  (void)top_k;

  auto eval_dataset_loader =
      _data_factory.getDataLoader(data, /* include_mach_labels= */ true,
                                  dataset::DatasetShuffleConfig(), verbose);

  return _classifier->evaluate(eval_dataset_loader, getMetrics(metrics, "val_"),
                               sparse_inference, verbose);
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

  return py::cast(predictImpl({sample}, sparse_inference, top_k).at(0));
}

std::vector<std::vector<std::pair<uint32_t, double>>>
UDTMachClassifier::predictImpl(const MapInputBatch& samples,
                               bool sparse_inference,
                               std::optional<uint32_t> top_k) {
  auto outputs = _classifier->model()
                     ->forward(_data_factory.featurizeInputBatch(samples),
                               sparse_inference)
                     .at(0);

  uint32_t num_classes = _data_factory.machIndex()->numEntities();

  if (top_k && top_k > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to "
        "predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  uint32_t k = top_k.value_or(_default_top_k_to_return);

  uint32_t batch_size = outputs->batchSize();

  std::vector<std::vector<std::pair<uint32_t, double>>> predicted_entities(
      batch_size);
#pragma omp parallel for default(none) \
    shared(outputs, predicted_entities, k, batch_size) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs->getVector(i);
    auto predictions = _data_factory.machIndex()->decode(
        /* output = */ vector,
        /* top_k = */ k,
        /* num_buckets_to_eval = */ _num_buckets_to_eval);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
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

  return py::cast(predictImpl(samples, sparse_inference, top_k));
}

py::object UDTMachClassifier::predictHashes(
    const MapInput& sample, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  return py::cast(
      predictHashesImpl({sample}, sparse_inference, force_non_empty, num_hashes)
          .at(0));
}

py::object UDTMachClassifier::predictHashesBatch(
    const MapInputBatch& samples, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  return py::cast(predictHashesImpl(samples, sparse_inference, force_non_empty,
                                    num_hashes));
}

std::vector<std::vector<uint32_t>> UDTMachClassifier::predictHashesImpl(
    const MapInputBatch& samples, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  auto outputs = _classifier->model()
                     ->forward(_data_factory.featurizeInputBatch(samples),
                               sparse_inference)
                     .at(0);

  uint32_t k = num_hashes.value_or(_data_factory.machIndex()->numHashes());

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, k, force_non_empty)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    if (force_non_empty) {
      heap = _data_factory.machIndex()->topKNonEmptyBuckets(output, k);
    } else {
      heap = output.findKLargestActivations(k);
    }

    std::vector<uint32_t> hashes;
    while (!heap.empty()) {
      auto [_, active_neuron] = heap.top();
      hashes.push_back(active_neuron);
      heap.pop();
    }

    std::reverse(hashes.begin(), hashes.end());

    all_hashes[i] = hashes;
  }

  return all_hashes;
}

py::object UDTMachClassifier::outputCorrectness(
    const MapInputBatch& samples, const std::vector<uint32_t>& labels,
    bool sparse_inference, std::optional<uint32_t> num_hashes) {
  std::vector<std::vector<uint32_t>> top_buckets = predictHashesImpl(
      samples, sparse_inference, /* force_non_empty = */ true, num_hashes);

  std::vector<uint32_t> matching_buckets(labels.size());
  std::exception_ptr hashes_err;

#pragma omp parallel for default(none) \
    shared(labels, top_buckets, matching_buckets, hashes_err)
  for (uint32_t i = 0; i < labels.size(); i++) {
    try {
      std::vector<uint32_t> hashes =
          _data_factory.machIndex()->getHashes(labels[i]);
      uint32_t count = 0;
      for (auto hash : hashes) {
        if (std::count(top_buckets[i].begin(), top_buckets[i].end(), hash) >
            0) {
          count++;
        }
      }
      matching_buckets[i] = count;
    } catch (const std::exception& e) {
#pragma omp critical
      hashes_err = std::current_exception();
    }
  }

  if (hashes_err) {
    std::rethrow_exception(hashes_err);
  }

  return py::cast(matching_buckets);
}

void UDTMachClassifier::setModel(const ModelPtr& model) {
  bolt::nn::model::ModelPtr& curr_model = _classifier->model();

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
    const bolt::train::DistributedCommPtr& comm) {
  thirdai::data::LoaderPtr val_dataset_loader;
  if (val_data) {
    val_dataset_loader =
        _data_factory.getDataLoader(val_data, /* include_mach_labels= */ true,
                                    options.shuffle_config, options.verbose);
  }

  addBalancingSamples(data, strong_column_names, weak_column_names);

  auto train_dataset_loader = _data_factory.getColdStartDataLoader(
      data, strong_column_names, weak_column_names,
      /* include_mach_labels= */ true, /* fast_approximation= */ false,
      options.shuffle_config, options.verbose);

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            getMetrics(train_metrics, "train_"),
                            val_dataset_loader, getMetrics(val_metrics, "val_"),
                            callbacks, options, comm);
}

py::object UDTMachClassifier::embedding(const MapInput& sample) {
  return _classifier->embedding(_data_factory.featurizeInput(sample));
}

uint32_t expectInteger(const Label& label) {
  if (!std::holds_alternative<uint32_t>(label)) {
    throw std::invalid_argument("Must use integer label.");
  }
  return std::get<uint32_t>(label);
}

py::object UDTMachClassifier::entityEmbedding(const Label& label) {
  std::vector<uint32_t> hashed_neurons =
      _data_factory.machIndex()->getHashes(expectInteger(label));

  auto outputs = _classifier->model()->outputs();

  if (outputs.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }
  auto fc = bolt::nn::ops::FullyConnected::cast(outputs.at(0)->op());
  if (!fc) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto fc_layer = fc->kernel();

  std::vector<float> averaged_embedding(fc_layer->getInputDim());
  for (uint32_t neuron_id : hashed_neurons) {
    auto weights = fc_layer->getWeightsByNeuron(neuron_id);
    if (weights.size() != averaged_embedding.size()) {
      throw std::invalid_argument("Output dim mismatch.");
    }
    for (uint32_t i = 0; i < weights.size(); i++) {
      averaged_embedding[i] += weights[i];
    }
  }

  // TODO(david) try averaging and summing
  for (float& weight : averaged_embedding) {
    weight /= averaged_embedding.size();
  }

  NumpyArray<float> np_weights(averaged_embedding.size());

  std::copy(averaged_embedding.begin(), averaged_embedding.end(),
            np_weights.mutable_data());

  return std::move(np_weights);
}

void UDTMachClassifier::updateSamplingStrategy() {
  auto mach_index = _data_factory.machIndex();

  auto output_layer = bolt::nn::ops::FullyConnected::cast(
      _classifier->model()->opExecutionOrder().back());

  const auto& neuron_index = output_layer->kernel()->neuronIndex();

  float index_sparsity = mach_index->sparsity();
  if (index_sparsity > 0 && index_sparsity <= _mach_sampling_threshold) {
    // TODO(Nicholas) add option to specify new neuron index in set sparsity.
    output_layer->setSparsity(index_sparsity, false, false);
    auto new_index = bolt::nn::MachNeuronIndex::make(mach_index);
    output_layer->kernel()->setNeuronIndex(new_index);

  } else {
    if (std::dynamic_pointer_cast<bolt::nn::MachNeuronIndex>(neuron_index)) {
      float sparsity = utils::autotuneSparsity(mach_index->numBuckets());

      auto sampling_config = bolt::DWTASamplingConfig::autotune(
          mach_index->numBuckets(), sparsity,
          /* experimental_autotune= */ false);

      output_layer->setSparsity(sparsity, false, false);

      if (sampling_config) {
        auto new_index = sampling_config->getNeuronIndex(
            output_layer->dim(), output_layer->inputDim());
        output_layer->kernel()->setNeuronIndex(new_index);
      }
    }
  }
}

void UDTMachClassifier::introduceDocuments(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool fast_approximation, bool verbose) {
  auto dataset_loader = _data_factory.getColdStartDataLoader(
      data_source, strong_column_names, weak_column_names,
      /* include_mach_labels= */ false,
      /* fast_approximation= */ fast_approximation,
      dataset::DatasetShuffleConfig(), verbose);

  auto [data, labels] = dataset_loader->all(defaults::BATCH_SIZE);

  uint32_t num_buckets_to_sample = num_buckets_to_sample_opt.value_or(
      _data_factory.machIndex()->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  bolt::train::python::CtrlCCheck ctrl_c_check;

  for (size_t batch_idx = 0; batch_idx < data.size(); batch_idx++) {
    // Note: using sparse inference here could cause issues because the
    // mach index sampler will only return nonempty buckets, which could
    // cause new docs to only be mapped to buckets already containing
    // entities.
    auto scores = _classifier->model()->forward(data.at(batch_idx)).at(0);

    for (uint32_t i = 0; i < scores->batchSize(); i++) {
      const BoltVector& label = labels.at(batch_idx).at(0)->getVector(0);

      uint32_t label_id = label.active_neurons[0];
      top_k_per_doc[label_id].push_back(
          scores->getVector(i).findKLargestActivations(num_buckets_to_sample));
    }

    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                  num_random_hashes);
    _data_factory.machIndex()->insert(doc, hashes);

    ctrl_c_check();
  }

  addBalancingSamples(data_source, strong_column_names, weak_column_names);

  updateSamplingStrategy();
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes) {
  auto input = _data_factory.featurizeInputColdStart(
      document, strong_column_names, weak_column_names);

  introduceLabelHelper(input, new_label, num_buckets_to_sample,
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

std::vector<uint32_t> UDTMachClassifier::topHashesForDoc(
    std::vector<TopKActivationsQueue>&& top_k_per_sample,
    uint32_t num_buckets_to_sample, uint32_t num_random_hashes) const {
  const auto& mach_index = _data_factory.machIndex();

  uint32_t num_hashes = mach_index->numHashes();

  if (num_buckets_to_sample < mach_index->numHashes()) {
    throw std::invalid_argument(
        "Sampling from fewer buckets than num_hashes is not supported. If "
        "you'd like to introduce using fewer hashes, please reset the number "
        "of hashes for the index.");
  }

  if (num_buckets_to_sample > mach_index->numBuckets()) {
    throw std::invalid_argument(
        "Cannot sample more buckets than there are in the index.");
  }

  std::unordered_map<uint32_t, BucketScore> hash_freq_and_scores;
  for (auto& top_k : top_k_per_sample) {
    while (!top_k.empty()) {
      auto [activation, active_neuron] = top_k.top();
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, activation};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += activation;
      }
      top_k.pop();
    }
  }

  // We sort the hashes first by number of occurances and tiebreak with
  // the higher aggregated score if necessary. We don't only use the
  // activations since those typically aren't as useful as the
  // frequencies.
  std::vector<std::pair<uint32_t, BucketScore>> sorted_hashes(
      hash_freq_and_scores.begin(), hash_freq_and_scores.end());

  CompareBuckets cmp;
  std::sort(sorted_hashes.begin(), sorted_hashes.end(), cmp);

  if (num_buckets_to_sample > num_hashes) {
    // If we are sampling more buckets then we end up using we rerank the
    // buckets based on size to load balance the index.
    std::sort(sorted_hashes.begin(),
              sorted_hashes.begin() + num_buckets_to_sample,
              [&mach_index, &cmp](const auto& lhs, const auto& rhs) {
                size_t lhs_size = mach_index->bucketSize(lhs.first);
                size_t rhs_size = mach_index->bucketSize(rhs.first);

                // Give preference to emptier buckets. If buckets are
                // equally empty, use one with the best score.
                if (lhs_size == rhs_size) {
                  return cmp(lhs, rhs);
                }

                return lhs_size < rhs_size;
              });
  }

  std::vector<uint32_t> new_hashes;

  // We can optionally specify the number of hashes we'd like to be
  // random for a new document. This is to encourage an even distribution
  // among buckets.
  if (num_random_hashes > num_hashes) {
    throw std::invalid_argument(
        "num_random_hashes cannot be greater than num hashes.");
  }

  uint32_t num_informed_hashes = num_hashes - num_random_hashes;
  for (uint32_t i = 0; i < num_informed_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes.push_back(hash);
  }

  uint32_t num_buckets = _data_factory.machIndex()->numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, num_buckets - 1);
  std::mt19937 rand(global_random::nextSeed());

  for (uint32_t i = 0; i < num_random_hashes; i++) {
    new_hashes.push_back(int_dist(rand));
  }

  return new_hashes;
}

void UDTMachClassifier::introduceLabel(
    const MapInputBatch& samples, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  introduceLabelHelper(_data_factory.featurizeInputBatch(samples), new_label,
                       num_buckets_to_sample_opt, num_random_hashes);
}

void UDTMachClassifier::introduceLabelHelper(
    const TensorList& samples, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
  auto output = _classifier->model()->forward(samples).at(0);

  uint32_t num_buckets_to_sample = num_buckets_to_sample_opt.value_or(
      _data_factory.machIndex()->numHashes());

  std::vector<TopKActivationsQueue> top_ks;
  for (uint32_t i = 0; i < output->batchSize(); i++) {
    top_ks.push_back(
        output->getVector(i).findKLargestActivations(num_buckets_to_sample));
  }

  auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                num_random_hashes);

  _data_factory.machIndex()->insert(expectInteger(new_label), hashes);

  updateSamplingStrategy();
}

void UDTMachClassifier::forget(const Label& label) {
  _data_factory.machIndex()->erase(expectInteger(label));

  if (_data_factory.machIndex()->numEntities() == 0) {
    std::cout << "Warning. Every learned class has been forgotten. The model "
                 "will currently return nothing on calls to evaluate, "
                 "predict, or predictBatch."
              << std::endl;
  }

  updateSamplingStrategy();
}

void UDTMachClassifier::addBalancingSamples(
    const dataset::DataSourcePtr& data_source,
    const std::vector<std::string>& strong_columns,
    const std::vector<std::string>& weak_columns) {
  if (_rlhf_sampler) {
    data_source->restart();

    thirdai::data::LoaderPtr loader;
    if (strong_columns.empty() && weak_columns.empty()) {
      loader = _data_factory.getDataLoader(data_source,
                                           /* include_mach_labels= */ true,
                                           dataset::DatasetShuffleConfig(),
                                           /* verbose= */ false);
    } else {
      loader = _data_factory.getColdStartDataLoader(
          data_source, strong_columns, weak_columns,
          /* include_mach_labels= */ true, /* fast_approximation= */ false,
          dataset::DatasetShuffleConfig(), /* verbose= */ false);
    }
    auto [data, labels] =
        loader
            ->next(/* batch_size= */ defaults::MAX_BALANCING_SAMPLES,
                   /* max_batches= */ 1)
            .value();

    auto input_tensor = data.at(0).at(0);
    auto label_tensor = labels.at(0).at(0);
    auto doc_id_tensor = labels.at(0).at(1);

    for (uint32_t i = 0; i < input_tensor->batchSize(); i++) {
      const BoltVector& doc_id_vec = doc_id_tensor->getVector(i);
      if (doc_id_vec.len < 1) {
        continue;
      }

      uint32_t doc_id = doc_id_vec.active_neurons[0];

      const BoltVector& input = input_tensor->getVector(i);
      const BoltVector& label = label_tensor->getVector(i);
      _rlhf_sampler->addSample(doc_id, input, label);
    }
    data_source->restart();
  }
}

void UDTMachClassifier::requireRLHFSampler() {
  if (!_rlhf_sampler) {
    throw std::runtime_error(
        "This model was not configured to support rlhf. Please pass "
        "{'rlhf': "
        "True} in the model options or call enable_rlhf().");
  }
}

BoltVector makeLabelFromHashes(const std::vector<uint32_t>& hashes,
                               uint32_t n_buckets, std::mt19937& rng) {
  std::vector<uint32_t> indices;
  std::sample(hashes.begin(), hashes.end(), std::back_inserter(indices),
              n_buckets, rng);

  return BoltVector::makeSparseVector(indices,
                                      std::vector<float>(indices.size(), 1.0));
}

void UDTMachClassifier::associate(
    const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
    uint32_t n_buckets, uint32_t n_association_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
  auto teaching_samples = getAssociateSamples(source_target_samples);

  teach(teaching_samples, n_buckets, n_association_samples, n_balancing_samples,
        learning_rate, epochs);
}

void UDTMachClassifier::upvote(
    const std::vector<std::pair<MapInput, uint32_t>>& source_target_samples,
    uint32_t n_upvote_samples, uint32_t n_balancing_samples,
    float learning_rate, uint32_t epochs) {
  std::vector<std::pair<MapInput, std::vector<uint32_t>>> teaching_samples;
  teaching_samples.reserve(source_target_samples.size());
  for (const auto& [source, target] : source_target_samples) {
    teaching_samples.emplace_back(source,
                                  _data_factory.machIndex()->getHashes(target));
  }
  uint32_t n_buckets = _data_factory.machIndex()->numHashes();
  teach(teaching_samples, n_buckets, n_upvote_samples, n_balancing_samples,
        learning_rate, epochs);
}

void UDTMachClassifier::teach(
    const std::vector<std::pair<MapInput, std::vector<uint32_t>>>&
        source_target_samples,
    uint32_t n_buckets, uint32_t n_teaching_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
  requireRLHFSampler();

  auto samples = _rlhf_sampler->balancingSamples(n_balancing_samples *
                                                 source_target_samples.size());

  std::mt19937 rng(global_random::nextSeed());

  for (const auto& [source, target_hashes] : source_target_samples) {
    BoltVector source_vec =
        _data_factory.featurizeInput(source).at(0)->getVector(0);
    for (uint32_t i = 0; i < n_teaching_samples; i++) {
      samples.emplace_back(source_vec,
                           makeLabelFromHashes(target_hashes, n_buckets, rng));
    }
  }

  std::shuffle(samples.begin(), samples.end(), rng);

  std::vector<
      std::pair<bolt::nn::tensor::TensorList, bolt::nn::tensor::TensorList>>
      batches;

  uint32_t input_dim = _classifier->model()->inputDims().at(0);
  uint32_t label_dim = _classifier->model()->labelDims().at(0);
  uint32_t batch_size = defaults::ASSOCIATE_BATCH_SIZE;

  for (size_t i = 0; i < samples.size(); i += batch_size) {
    std::vector<BoltVector> inputs;
    std::vector<BoltVector> labels;

    size_t batch_end = std::min(i + batch_size, samples.size());
    for (size_t j = i; j < batch_end; j++) {
      inputs.emplace_back(std::move(samples[j].first));
      labels.emplace_back(std::move(samples[j].second));
    }

    auto input_tensor = bolt::nn::tensor::Tensor::convert(
        BoltBatch(std::move(inputs)), input_dim);

    auto label_tensor = bolt::nn::tensor::Tensor::convert(
        BoltBatch(std::move(labels)), label_dim);

    batches.push_back(
        {{input_tensor},
         {label_tensor, placeholderDocIds(label_tensor->batchSize())}});
  }

  for (uint32_t i = 0; i < epochs; i++) {
    for (const auto& [x, y] : batches) {
      _classifier->model()->trainOnBatch(x, y);
      _classifier->model()->updateParameters(learning_rate);
    }
  }
}

std::vector<std::pair<MapInput, std::vector<uint32_t>>>
UDTMachClassifier::getAssociateSamples(
    const std::vector<std::pair<MapInput, MapInput>>& source_target_samples) {
  MapInputBatch batch;
  for (const auto& [_, target] : source_target_samples) {
    batch.emplace_back(target);
  }

  auto all_predicted_hashes = predictHashesImpl(
      batch, /* sparse_inference = */ false, /* force_non_empty = */ true);

  std::vector<std::pair<MapInput, std::vector<uint32_t>>> associate_samples;
  associate_samples.reserve(source_target_samples.size());
  for (uint32_t i = 0; i < source_target_samples.size(); i++) {
    associate_samples.emplace_back(source_target_samples[i].first,
                                   all_predicted_hashes[i]);
  }

  return associate_samples;
}

py::object UDTMachClassifier::associateTrain(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto dataset = _data_factory.getDataLoader(
      balancing_data, /* include_mach_labels= */ true,
      dataset::DatasetShuffleConfig(), options.verbose);

  return associateTrainHelper(dataset, source_target_samples, n_buckets,
                              n_association_samples, learning_rate, epochs,
                              metrics, options);
}

py::object UDTMachClassifier::associateColdStart(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto dataset = _data_factory.getColdStartDataLoader(
      balancing_data, strong_column_names, weak_column_names,
      /* include_mach_labels= */ true, /* fast_approximation= */ false,
      options.shuffle_config, options.verbose);

  return associateTrainHelper(dataset, source_target_samples, n_buckets,
                              n_association_samples, learning_rate, epochs,
                              metrics, options);
}

py::object UDTMachClassifier::associateTrainHelper(
    const thirdai::data::LoaderPtr& balancing_data,
    const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  warnOnNonHashBasedMetrics(metrics);

  auto associate_samples = getAssociateSamples(source_target_samples);

  std::mt19937 rng(global_random::nextSeed());

  std::vector<MapInput> inputs;
  std::vector<std::vector<uint32_t>> target_hashes;

  for (auto& [input, hashes] : associate_samples) {
    BoltVector input_vec =
        _data_factory.featurizeInput(input).at(0)->getVector(0);
    for (uint32_t i = 0; i < n_association_samples; i++) {
      inputs.push_back(input);

      std::vector<uint32_t> selected_hashes;
      std::sample(hashes.begin(), hashes.end(),
                  std::back_inserter(selected_hashes), n_buckets, rng);
      target_hashes.push_back(selected_hashes);
    }
  }

  auto columns = thirdai::data::ColumnMap::fromMapInputBatch(inputs);
  columns = _data_factory.applyInputTransformation(std::move(columns));
  columns.setColumn(
      _data_factory.featurizedMachLabelColumn(),
      thirdai::data::ArrayColumn<uint32_t>::make(
          std::move(target_hashes), _data_factory.machIndex()->numBuckets()));
  columns.setColumn(_data_factory.featurizedEntityIdColumn(),
                    thirdai::data::ValueColumn<uint32_t>::make(
                        std::vector<uint32_t>(columns.numRows(), 0),
                        std::numeric_limits<uint32_t>::max()));
  columns.setColumn(_data_factory.coldStartLabelColumn(),
                    thirdai::data::ValueColumn<std::string>::make(
                        std::vector<std::string>(columns.numRows())));

  balancing_data->addToShuffleBuffer(std::move(columns));

  return _classifier->train(balancing_data, learning_rate, epochs,
                            getMetrics(metrics, "train_"),
                            /* val_dataset */ nullptr, /* val_metrics= */ {},
                            /* callbacks= */ {}, options, /*comm= */ nullptr);
}

void UDTMachClassifier::setDecodeParams(uint32_t top_k_to_return,
                                        uint32_t num_buckets_to_eval) {
  if (top_k_to_return == 0 || num_buckets_to_eval == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  uint32_t num_buckets = _data_factory.machIndex()->numBuckets();
  if (num_buckets_to_eval > num_buckets) {
    throw std::invalid_argument(
        "Cannot eval with num_buckets_to_eval greater than " +
        std::to_string(num_buckets) + ".");
  }

  uint32_t num_classes = _data_factory.machIndex()->numEntities();
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

void UDTMachClassifier::setIndex(const dataset::mach::MachIndexPtr& index) {
  // block allows indexes with different number of hashes but not num buckets
  _data_factory.setIndex(index);

  updateSamplingStrategy();
}

void UDTMachClassifier::setMachSamplingThreshold(float threshold) {
  _mach_sampling_threshold = threshold;
  updateSamplingStrategy();
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

  bolt::nn::autograd::ComputationPtr output = model->outputs().front();
  bolt::nn::autograd::ComputationPtr hash_labels = model->labels().front();
  bolt::nn::autograd::ComputationPtr true_class_labels = model->labels().back();
  bolt::nn::loss::LossPtr loss = model->losses().front();

  InputMetrics metrics;
  for (const auto& name : metric_names) {
    if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 10, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachPrecision>(
          _data_factory.machIndex(), _num_buckets_to_eval, output,
          true_class_labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachRecall>(
          _data_factory.machIndex(), _num_buckets_to_eval, output,
          true_class_labels, k);
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

bolt::nn::tensor::TensorPtr UDTMachClassifier::placeholderDocIds(
    uint32_t batch_size) {
  return bolt::nn::tensor::Tensor::sparse(batch_size,
                                          std::numeric_limits<uint32_t>::max(),
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
  archive(cereal::base_class<UDTBackend>(this), _classifier, _data_factory,
          _default_top_k_to_return, _num_buckets_to_eval,
          _mach_sampling_threshold, _rlhf_sampler);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)