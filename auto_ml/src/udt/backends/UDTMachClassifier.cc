#include "UDTMachClassifier.h"
#include <cereal/types/optional.hpp>
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/embedding_prototype/TextEmbeddingModel.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/Validation.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/mach/MachBlock.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/StringManipulation.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
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
    const config::ArgumentMap& user_args)
    : _min_num_eval_results(defaults::MACH_MIN_NUM_EVAL_RESULTS),
      _top_k_per_eval_aggregation(defaults::MACH_TOP_K_PER_EVAL_AGGREGATION) {
  uint32_t num_buckets = user_args.get<uint32_t>(
      "extreme_output_dim", "integer", autotuneMachOutputDim(n_target_classes));
  uint32_t num_hashes = user_args.get<uint32_t>(
      "extreme_num_hashes", "integer",
      autotuneMachNumHashes(n_target_classes, num_buckets));

  _classifier = utils::Classifier::make(
      utils::buildModel(
          /* input_dim= */ tabular_options.feature_hash_range,
          /* output_dim= */ num_buckets,
          /* args= */ user_args, /* model_config= */ model_config,
          /* use_sigmoid_bce = */ true),
      user_args.get<bool>("freeze_hash_tables", "boolean",
                          defaults::FREEZE_HASH_TABLES));

  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart?

  dataset::mach::MachIndexPtr mach_index;
  if (integer_target) {
    mach_index = dataset::mach::MachIndex::make(
        /* num_buckets = */ num_buckets, /* num_hashes = */ num_hashes,
        /* num_elements = */ n_target_classes);
  } else {
    throw std::invalid_argument(
        "Option 'integer_target=True' must be specified when using extreme "
        "classification options.");
  }

  _mach_label_block = dataset::mach::MachBlock::make(target_name, mach_index,
                                                     target_config->delimiter);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = data::TabularDatasetFactory::make(
      /* input_data_types = */ input_data_types,
      /* provided_temporal_relationships = */ temporal_tracking_relationships,
      /* label_blocks = */ {dataset::BlockList({_mach_label_block})},
      /* label_col_names = */ std::set<std::string>{target_name},
      /* options = */ tabular_options, /* force_parallel = */ force_parallel);

  // No limit on the number of classes.
  auto doc_id_block = dataset::NumericalCategoricalBlock::make(
      target_name, std::numeric_limits<uint32_t>::max());

  _hashes_and_doc_id_factory = data::TabularDatasetFactory::make(
      /* input_data_types = */ input_data_types,
      /* provided_temporal_relationships = */ temporal_tracking_relationships,
      /* label_blocks = */
      {dataset::BlockList({_mach_label_block}),
       dataset::BlockList({doc_id_block})},
      /* label_col_names = */ std::set<std::string>{target_name},
      /* options = */ tabular_options, /* force_parallel = */ force_parallel);

  auto hash_processing_block = dataset::NumericalCategoricalBlock::make(
      /* col= */ target_name,
      /* n_classes= */ num_buckets,
      /* delimiter= */ ' ',
      /* normalize_categories= */ false);

  // We want to be able to train input samples on a specific set of hashes so
  // we create a separate dataset factory that does all the same things as the
  // regular dataset factory except with the label block switched out
  _pre_hashed_labels_dataset_factory = data::TabularDatasetFactory::make(
      /* input_data_types = */ input_data_types,
      /* provided_temporal_relationships = */ temporal_tracking_relationships,
      /* label_blocks = */ {dataset::BlockList({hash_processing_block})},
      /* label_col_names = */ std::set<std::string>{target_name},
      /* options = */ tabular_options, /* force_parallel = */ force_parallel);

  _sparse_inference_threshold =
      user_args.get<float>("sparse_inference_threshold", "float",
                           defaults::MACH_SPARSE_INFERENCE_THRESHOLD);

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
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<CallbackPtr>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  ValidationDatasetLoader validation_dataset_loader;
  if (validation) {
    validation_dataset_loader =
        ValidationDatasetLoader(_dataset_factory->getLabeledDatasetLoader(
                                    validation->first, /* shuffle= */ false),
                                validation->second);
  }

  addBalancingSamples(data);

  auto train_dataset_loader =
      _dataset_factory->getLabeledDatasetLoader(data, /* shuffle= */ true);

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            validation_dataset_loader, batch_size_opt,
                            max_in_memory_batches, metrics, callbacks, verbose,
                            logging_interval);
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose,
                                       std::optional<uint32_t> top_k) {
  (void)top_k;

  auto eval_dataset_loader =
      _dataset_factory->getLabeledDatasetLoader(data, /* shuffle= */ false);

  // TODO(david) eventually we should use backend specific metrics

  return _classifier->evaluate(eval_dataset_loader, metrics, sparse_inference,
                               verbose);
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class,
                                      std::optional<uint32_t> top_k) {
  (void)top_k;
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  return py::cast(predictImpl(sample, sparse_inference));
}

std::vector<std::pair<uint32_t, double>> UDTMachClassifier::predictImpl(
    const MapInput& sample, bool sparse_inference) {
  auto outputs = _classifier->model()->forward(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  const BoltVector& output = outputs.at(0)->getVector(0);

  auto decoded_output = _mach_label_block->index()->decode(
      /* output = */ output,
      /* min_num_eval_results = */ _min_num_eval_results,
      /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);

  return decoded_output;
}

py::object UDTMachClassifier::trainBatch(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] = _dataset_factory->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class,
                                           std::optional<uint32_t> top_k) {
  (void)top_k;
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  auto outputs = _classifier->model()
                     ->forward(_dataset_factory->featurizeInputBatch(samples),
                               sparse_inference)
                     .at(0);

  std::vector<std::vector<std::pair<uint32_t, double>>> predicted_entities(
      outputs->batchSize());
#pragma omp parallel for default(none) shared(outputs, predicted_entities)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& vector = outputs->getVector(i);
    auto predictions = _mach_label_block->index()->decode(
        /* output = */ vector,
        /* min_num_eval_results = */ _min_num_eval_results,
        /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);
    predicted_entities[i] = predictions;
  }

  return py::cast(predicted_entities);
}

py::object UDTMachClassifier::trainWithHashes(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] =
      _pre_hashed_labels_dataset_factory->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::predictHashes(const MapInput& sample,
                                            bool sparse_inference) {
  return py::cast(predictHashesImpl(sample, sparse_inference));
}

std::vector<uint32_t> UDTMachClassifier::predictHashesImpl(
    const MapInput& sample, bool sparse_inference) {
  auto outputs = _classifier->model()->forward(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  const BoltVector& output = outputs.at(0)->getVector(0);

  uint32_t k = _mach_label_block->index()->numHashes();
  auto heap = output.findKLargestActivations(k);

  std::vector<uint32_t> hashes_to_return;
  while (hashes_to_return.size() < k && !heap.empty()) {
    auto [_, active_neuron] = heap.top();
    hashes_to_return.push_back(active_neuron);
    heap.pop();
  }

  std::reverse(hashes_to_return.begin(), hashes_to_return.end());

  return hashes_to_return;
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
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<ValidationDataSource>& validation,
    const std::vector<CallbackPtr>& callbacks,
    std::optional<size_t> max_in_memory_batches, bool verbose) {
  auto metadata = getColdStartMetaData();

  auto data_source = cold_start::preprocessColdStartTrainSource(
      data, strong_column_names, weak_column_names, _dataset_factory, metadata);

  return train(data_source, learning_rate, epochs, validation,
               /* batch_size_opt = */ std::nullopt,
               /* max_in_memory_batches= */ max_in_memory_batches, metrics,
               /* callbacks= */ callbacks, /* verbose= */ verbose,
               /* logging_interval= */ std::nullopt);
}

py::object UDTMachClassifier::embedding(const MapInput& sample) {
  return _classifier->embedding(_dataset_factory->featurizeInput(sample));
}

uint32_t expectInteger(const Label& label) {
  if (!std::holds_alternative<uint32_t>(label)) {
    throw std::invalid_argument("Must use integer label.");
  }
  return std::get<uint32_t>(label);
}

py::object UDTMachClassifier::entityEmbedding(const Label& label) {
  std::vector<uint32_t> hashed_neurons =
      _mach_label_block->index()->getHashes(expectInteger(label));

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

std::string UDTMachClassifier::textColumnForDocumentIntroduction() {
  if (_dataset_factory->inputDataTypes().size() != 1 ||
      !data::asText(_dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "Introducing documents can only be used when UDT is configured with a "
        "single text input column and target column. The current model is "
        "configured with " +
        std::to_string(_dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }

  return _dataset_factory->inputDataTypes().begin()->first;
}

void UDTMachClassifier::updateSamplingStrategy() {
  auto mach_index = _mach_label_block->index();

  auto output_layer = bolt::nn::ops::FullyConnected::cast(
      _classifier->model()->opExecutionOrder().back());

  const auto& neuron_index = output_layer->kernel()->neuronIndex();

  float index_sparsity = mach_index->sparsity();
  if (index_sparsity > 0 && index_sparsity <= _sparse_inference_threshold) {
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
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt) {
  auto metadata = getColdStartMetaData();

  auto cold_start_data = cold_start::preprocessColdStartTrainSource(
      data, strong_column_names, weak_column_names, _dataset_factory, metadata);

  auto dataset_loader =
      _dataset_factory->getUnLabeledDatasetLoader(cold_start_data);

  auto doc_samples = dataset_loader->loadAll(defaults::BATCH_SIZE);

  auto doc_samples_tensors = bolt::train::convertDatasets(
      doc_samples, _classifier->model()->inputDims());

  const auto& labels = cold_start_data->labelColumn();
  uint32_t row_idx = 0;

  uint32_t num_buckets_to_sample = num_buckets_to_sample_opt.value_or(
      _mach_label_block->index()->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  for (const auto& batch : doc_samples_tensors) {
    // Note: using sparse inference here could cause issues because the mach
    // index sampler will only return nonempty buckets, which could cause new
    // docs to only be mapped to buckets already containing entities.
    auto scores = _classifier->model()->forward(batch).at(0);

    for (uint32_t i = 0; i < scores->batchSize(); i++) {
      uint32_t label = std::stoi((*labels)[row_idx++]);
      top_k_per_doc[label].push_back(
          scores->getVector(i).findKLargestActivations(num_buckets_to_sample));
    }
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample);
    _mach_label_block->index()->insert(doc, hashes);
  }

  addBalancingSamples(cold_start_data);

  updateSamplingStrategy();
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample) {
  std::string text_column_name = textColumnForDocumentIntroduction();

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ _mach_label_block->columnName(),
      /* output_column_name= */
      text_column_name);

  MapInputBatch batch;
  for (const auto& row : augmentation.augmentMapInput(document)) {
    MapInput input = {{text_column_name, row}};
    batch.push_back(input);
  }

  introduceLabel(batch, new_label, num_buckets_to_sample);
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
    uint32_t num_buckets_to_sample) const {
  const auto& mach_index = _mach_label_block->index();

  uint32_t num_hashes = mach_index->numHashes();

  if (num_buckets_to_sample < mach_index->numHashes()) {
    std::cout << "Warning. Sampling from fewer buckets than num_hashes. "
                 "Defaulting to sampling from num_hashes buckets.";
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

  // We sort the hashes first by number of occurances and tiebreak with the
  // higher aggregated score if necessary. We don't only use the activations
  // since those typically aren't as useful as the frequencies.
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

                // Give preference to emptier buckets. If buckets are equally
                // empty, use one with the best score.
                if (lhs_size == rhs_size) {
                  return cmp(lhs, rhs);
                }

                return lhs_size < rhs_size;
              });
  }

  std::vector<uint32_t> new_hashes(num_hashes);
  for (uint32_t i = 0; i < num_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes[i] = hash;
  }

  return new_hashes;
}

void UDTMachClassifier::introduceLabel(
    const MapInputBatch& samples, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample_opt) {
  // Note: using sparse inference here could cause issues because the mach
  // index sampler will only return nonempty buckets, which could cause new
  // docs to only be mapped to buckets already containing entities.
  auto output = _classifier->model()
                    ->forward(_dataset_factory->featurizeInputBatch(samples),
                              /* use_sparsity = */ false)
                    .at(0);

  uint32_t num_buckets_to_sample = num_buckets_to_sample_opt.value_or(
      _mach_label_block->index()->numHashes());

  std::vector<TopKActivationsQueue> top_ks;
  for (uint32_t i = 0; i < output->batchSize(); i++) {
    top_ks.push_back(
        output->getVector(i).findKLargestActivations(num_buckets_to_sample));
  }

  auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample);

  _mach_label_block->index()->insert(expectInteger(new_label), hashes);

  updateSamplingStrategy();
}

void UDTMachClassifier::forget(const Label& label) {
  _mach_label_block->index()->erase(expectInteger(label));

  if (_mach_label_block->index()->numEntities() == 0) {
    std::cout << "Warning. Every learned class has been forgotten. The model "
                 "will currently return nothing on calls to evaluate, "
                 "predict, or predictBatch."
              << std::endl;
  }

  updateSamplingStrategy();
}

void UDTMachClassifier::addBalancingSamples(
    const dataset::DataSourcePtr& data) {
  if (_rlhf_sampler) {
    data->restart();
    auto samples =
        _hashes_and_doc_id_factory
            ->getLabeledDatasetLoader(data, /* shuffle= */ true)
            ->loadSome(/* batch_size= */ defaults::MAX_BALANCING_SAMPLES,
                       /* num_batches= */ 1, /* verbose= */ false)
            .value();

    for (uint32_t i = 0; i < samples.front()->len(); i++) {
      const BoltVector& doc_id_vec = samples.at(2)->at(0)[i];
      if (doc_id_vec.len != 1) {
        throw std::runtime_error("Expected doc id to be a single integer.");
      }
      uint32_t doc_id = samples.at(2)->at(0)[i].active_neurons[0];

      const BoltVector& input = samples.at(0)->at(0)[i];
      const BoltVector& label = samples.at(1)->at(0)[i];
      _rlhf_sampler->addSample(doc_id, input, label);
    }
    data->restart();
  }
}

void UDTMachClassifier::requireRLHFSampler() {
  if (!_rlhf_sampler) {
    throw std::runtime_error(
        "This model was not configured to support rlhf. Please pass {'rlhf': "
        "True} in the model options.");
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
  std::vector<std::pair<MapInput, std::vector<uint32_t>>> teaching_samples;
  teaching_samples.reserve(source_target_samples.size());
  for (const auto& [source, target] : source_target_samples) {
    teaching_samples.emplace_back(source, predictHashesImpl(target, false));
  }
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
    teaching_samples.emplace_back(
        source, _mach_label_block->index()->getHashes(target));
  }
  uint32_t n_buckets = _mach_label_block->index()->numHashes();
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
        _dataset_factory->featurizeInput(source).at(0)->getVector(0);
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

    batches.push_back({{input_tensor}, {label_tensor}});
  }

  for (uint32_t i = 0; i < epochs; i++) {
    for (const auto& [x, y] : batches) {
      _classifier->model()->trainOnBatch(x, y);
      _classifier->model()->updateParameters(learning_rate);
    }
  }
}

void UDTMachClassifier::setDecodeParams(uint32_t min_num_eval_results,
                                        uint32_t top_k_per_eval_aggregation) {
  if (min_num_eval_results == 0 || top_k_per_eval_aggregation == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  uint32_t num_buckets = _mach_label_block->index()->numBuckets();
  if (top_k_per_eval_aggregation > num_buckets) {
    throw std::invalid_argument(
        "Cannot eval with top_k_per_eval_aggregation greater than " +
        std::to_string(num_buckets) + ".");
  }

  uint32_t num_classes = _mach_label_block->index()->numEntities();
  if (min_num_eval_results > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  _min_num_eval_results = min_num_eval_results;
  _top_k_per_eval_aggregation = top_k_per_eval_aggregation;
}

void UDTMachClassifier::setIndex(const dataset::mach::MachIndexPtr& index) {
  // block allows indexes with different number of hashes but not output
  // ranges
  _mach_label_block->setIndex(index);

  updateSamplingStrategy();
}

TextEmbeddingModelPtr UDTMachClassifier::getTextEmbeddingModel(
    float distance_cutoff) const {
  return createTextEmbeddingModel(_classifier->model(), _dataset_factory,
                                  distance_cutoff);
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
  archive(cereal::base_class<UDTBackend>(this), _classifier, _mach_label_block,
          _dataset_factory, _pre_hashed_labels_dataset_factory,
          _hashes_and_doc_id_factory, _min_num_eval_results,
          _top_k_per_eval_aggregation, _sparse_inference_threshold,
          _rlhf_sampler);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)