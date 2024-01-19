#include "UDTMachClassifier.h"
#include <wrappers/src/EigenDenseWrapper.h>
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/inference/EmbFcInference.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/MachPrecision.h>
#include <bolt/src/train/metrics/MachRecall.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachBlock.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/Version.h>
#include <utils/text/StringManipulation.h>
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

  if (!integer_target) {
    throw std::invalid_argument(
        "Option 'integer_target=True' must be specified when using extreme "
        "classification options.");
  }

  dataset::mach::MachIndexPtr mach_index = dataset::mach::MachIndex::make(
      /* num_buckets = */ num_buckets, /* num_hashes = */ num_hashes,
      /* num_elements = */ n_target_classes);

  _mach_label_block = dataset::mach::MachBlock::make(target_name, mach_index,
                                                     target_config->delimiter);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  // No limit on the number of classes.
  auto doc_id_block = dataset::NumericalCategoricalBlock::make(
      target_name, std::numeric_limits<uint32_t>::max(),
      /* delimiter= */ target_config->delimiter);

  _dataset_factory = TabularDatasetFactory::make(
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
  _pre_hashed_labels_dataset_factory = TabularDatasetFactory::make(
      /* input_data_types = */ input_data_types,
      /* provided_temporal_relationships = */ temporal_tracking_relationships,
      /* label_blocks = */ {dataset::BlockList({hash_processing_block})},
      /* label_col_names = */ std::set<std::string>{target_name},
      /* options = */ tabular_options, /* force_parallel = */ force_parallel);

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
    const bolt::DistributedCommPtr& comm) {
  dataset::DatasetLoaderPtr val_dataset_loader;
  if (val_data) {
    val_dataset_loader = _dataset_factory->getLabeledDatasetLoader(
        val_data, /* shuffle= */ false);
  }

  addBalancingSamples(data);

  auto train_dataset_loader = _dataset_factory->getLabeledDatasetLoader(
      data, /* shuffle= */ true, /* shuffle_config= */ options.shuffle_config);

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            getMetrics(train_metrics, "train_"),
                            val_dataset_loader, getMetrics(val_metrics, "val_"),
                            callbacks, options, comm);
}

py::object UDTMachClassifier::trainOnTensors(
    const bolt::LabeledDataset& train_data, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& train_metrics,
    const std::optional<bolt::LabeledDataset>& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options) {
  return _classifier->train(
      train_data, learning_rate, epochs, getMetrics(train_metrics, "train_"),
      val_data, getMetrics(val_metrics, "val_"), callbacks, options);
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

py::object UDTMachClassifier::trainWithHashes(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] =
      _pre_hashed_labels_dataset_factory->featurizeTrainingBatch(batch);
  labels.push_back(placeholderDocIds(batch.size()));

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
      _dataset_factory->getLabeledDatasetLoader(data, /* shuffle= */ false);

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
                     ->forward(_dataset_factory->featurizeInputBatch(samples),
                               sparse_inference)
                     .at(0);

  uint32_t num_classes = _mach_label_block->index()->numEntities();

  if (top_k && *top_k > num_classes) {
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
    auto predictions = _mach_label_block->index()->decode(
        /* output = */ vector,
        /* top_k = */ k,
        /* num_buckets_to_eval = */ _num_buckets_to_eval);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
}

py::object UDTMachClassifier::predictTensors(const bolt::TensorList& input_data,
                                             bool sparse_inference,
                                             std::optional<uint32_t> top_k) {
  auto outputs =
      _classifier->model()->forward(input_data, sparse_inference).at(0);

  uint32_t num_classes = _mach_label_block->index()->numEntities();

  if (top_k && *top_k > num_classes) {
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
    auto predictions = _mach_label_block->index()->decode(
        /* output = */ vector,
        /* top_k = */ k,
        /* num_buckets_to_eval = */ _num_buckets_to_eval);
    predicted_entities[i] = predictions;
  }

  return py::cast(predicted_entities);
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

py::object UDTMachClassifier::scoreBatch(
    const MapInputBatch& samples,
    const std::vector<std::vector<Label>>& classes,
    std::optional<uint32_t> top_k) {
  std::vector<std::unordered_set<uint32_t>> entities(classes.size());
  for (uint32_t row = 0; row < classes.size(); row++) {
    entities[row].reserve(classes[row].size());
    for (const auto& entity : classes[row]) {
      entities[row].insert(expectInteger(entity));
    }
  }

  // sparse inference could become an issue here because maybe the entities
  // we score wouldn't otherwise be in the top results, thus their buckets have
  // lower similarity and don't get selected by LSH
  auto outputs = _classifier->model()
                     ->forward(_dataset_factory->featurizeInputBatch(samples),
                               /* use_sparsity= */ false)
                     .at(0);

  size_t batch_size = samples.size();
  std::vector<std::vector<std::pair<uint32_t, double>>> scores(samples.size());

  const auto& index = _mach_label_block->index();
#pragma omp parallel for default(none) shared( \
    entities, outputs, scores, top_k, batch_size, index) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs->getVector(i);
    scores[i] = index->scoreEntities(vector, entities[i], top_k);
  }

  return py::cast(scores);
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
                     ->forward(_dataset_factory->featurizeInputBatch(samples),
                               sparse_inference)
                     .at(0);

  uint32_t k = num_hashes.value_or(_mach_label_block->index()->numHashes());

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, k, force_non_empty)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    if (force_non_empty) {
      heap = _mach_label_block->index()->topKNonEmptyBuckets(output, k);
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
          _mach_label_block->index()->getHashes(labels[i]);
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
  bolt::ModelPtr& curr_model = _classifier->model();

  utils::verifyCanSetModel(curr_model, model);

  curr_model = model;
}

py::object UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  auto metadata = getColdStartMetaData();

  if (!variable_length.has_value()) {
    auto data_source = cold_start::preprocessColdStartTrainSource(
        data, strong_column_names, weak_column_names, _dataset_factory,
        metadata, variable_length);

    return train(data_source, learning_rate, epochs, train_metrics, val_data,
                 val_metrics, callbacks, options, comm);
  }

  py::object history;
  for (uint32_t i = 0; i < epochs; i++) {
    auto data_source = cold_start::preprocessColdStartTrainSource(
        data, strong_column_names, weak_column_names, _dataset_factory,
        metadata, variable_length);

    history = train(data_source, learning_rate, /* epochs= */ 1, train_metrics,
                    val_data, val_metrics, callbacks, options, comm);
    data->restart();
    if (val_data) {
      val_data->restart();
    }
  }

  return history;
}

py::object UDTMachClassifier::embedding(const MapInputBatch& sample) {
  return _classifier->embedding(_dataset_factory->featurizeInputBatch(sample));
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
  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
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
      !asText(_dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "Introducing documents can only be used when UDT is configured "
        "with a "
        "single text input column and target column. The current model "
        "is "
        "configured with " +
        std::to_string(_dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }

  return _dataset_factory->inputDataTypes().begin()->first;
}

void UDTMachClassifier::updateSamplingStrategy() {
  auto mach_index = _mach_label_block->index();

  auto output_layer = bolt::FullyConnected::cast(
      _classifier->model()->opExecutionOrder().back());

  const auto& neuron_index = output_layer->kernel()->neuronIndex();

  float index_sparsity = mach_index->sparsity();
  if (index_sparsity > 0 && index_sparsity <= _mach_sampling_threshold) {
    // TODO(Nicholas) add option to specify new neuron index in set sparsity.
    output_layer->setSparsity(index_sparsity, false, false);
    auto new_index = bolt::MachNeuronIndex::make(mach_index);
    output_layer->kernel()->setNeuronIndex(new_index);

  } else {
    if (std::dynamic_pointer_cast<bolt::MachNeuronIndex>(neuron_index)) {
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

std::optional<bolt::EmbFcInference> inferenceModel(
    const bolt::ModelPtr& model) {
  auto computations = model->computationOrder();
  if (computations.size() != 3) {
    return std::nullopt;
  }

  auto input = std::dynamic_pointer_cast<bolt::Input>(computations.at(0)->op());
  auto emb = bolt::Embedding::cast(computations.at(1)->op());
  auto fc = bolt::FullyConnected::cast(computations.at(2)->op());

  // Model must be Input -> Embedding -> FullyConnected
  if (!input || !emb || !fc) {
    return std::nullopt;
  }
  // Currently this is only implmented for sigmoid activations for mach, but
  // could be extended in the future.
  if (fc->kernel()->getActivationFunction() !=
      bolt::ActivationFunction::Sigmoid) {
    return std::nullopt;
  }

  return std::make_optional<bolt::EmbFcInference>(emb, fc);
}

void UDTMachClassifier::introduceDocuments(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool fast_approximation, bool verbose) {
  auto metadata = getColdStartMetaData();

  dataset::cold_start::ColdStartDataSourcePtr cold_start_data;
  if (fast_approximation) {
    cold_start_data = cold_start::concatenatedDocumentDataSource(
        data, strong_column_names, weak_column_names, _dataset_factory,
        metadata);
  } else {
    cold_start_data = cold_start::preprocessColdStartTrainSource(
        data, strong_column_names, weak_column_names, _dataset_factory,
        metadata, /* variable_length= */ std::nullopt);
  }

  auto dataset_loader =
      _dataset_factory->getUnLabeledDatasetLoader(cold_start_data);

  auto doc_samples = dataset_loader->loadAll(defaults::BATCH_SIZE, verbose);

  auto doc_samples_tensors =
      bolt::convertDatasets(doc_samples, _classifier->model()->inputDims());

  const auto& labels = cold_start_data->labelColumn();
  uint32_t row_idx = 0;

  uint32_t num_buckets_to_sample = num_buckets_to_sample_opt.value_or(
      _mach_label_block->index()->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  auto inference_model = inferenceModel(_classifier->model());

  bolt::python::CtrlCCheck ctrl_c_check;

  for (const auto& batch : doc_samples_tensors) {
    // Note: using sparse inference here could cause issues because the
    // mach index sampler will only return nonempty buckets, which could
    // cause new docs to only be mapped to buckets already containing
    // entities.
    bolt::TensorPtr scores;
    if (inference_model) {
      scores = inference_model->forward(batch.at(0));
    } else {
      scores = _classifier->model()->forward(batch).at(0);
    }

    for (uint32_t i = 0; i < scores->batchSize(); i++) {
      uint32_t label = std::stoi(labels->value(row_idx++));
      top_k_per_doc[label].push_back(
          scores->getVector(i).findKLargestActivations(num_buckets_to_sample));
    }

    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                  num_random_hashes);
    _mach_label_block->index()->insert(doc, hashes);

    ctrl_c_check();
  }

  addBalancingSamples(cold_start_data);

  updateSamplingStrategy();
}

void UDTMachClassifier::introduceDocumentsAsVectors(
    const std::vector<std::vector<bolt::TensorPtr>>& input_tensors,
    const std::vector<uint32_t>& labels,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool verbose) {
  (void)verbose;
  uint32_t num_buckets_to_sample = num_buckets_to_sample_opt.value_or(
      _mach_label_block->index()->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  auto inference_model = inferenceModel(_classifier->model());

  bolt::python::CtrlCCheck ctrl_c_check;

  uint32_t row_idx = 0;

  for (const auto& batch : input_tensors) {
    // Note: using sparse inference here could cause issues because the
    // mach index sampler will only return nonempty buckets, which could
    // cause new docs to only be mapped to buckets already containing
    // entities.
    bolt::TensorPtr scores;
    if (inference_model) {
      scores = inference_model->forward(batch.at(0));
    } else {
      scores = _classifier->model()->forward(batch).at(0);
    }

    for (uint32_t i = 0; i < scores->batchSize(); i++) {
      uint32_t label = labels[row_idx++];
      top_k_per_doc[label].push_back(
          scores->getVector(i).findKLargestActivations(num_buckets_to_sample));
    }

    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                  num_random_hashes);
    _mach_label_block->index()->insert(doc, hashes);

    ctrl_c_check();
  }

  // addBalancingSamples(cold_start_data);

  updateSamplingStrategy();
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes) {
  std::string text_column_name = textColumnForDocumentIntroduction();

  data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* output_column_name= */ text_column_name);

  MapInputBatch batch;
  for (const auto& row : augmentation.augmentMapInput(document)) {
    MapInput input = {{text_column_name, row}};
    batch.push_back(input);
  }

  introduceLabel(batch, new_label, num_buckets_to_sample, num_random_hashes);
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
  const auto& mach_index = _mach_label_block->index();

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

  // We sort the hashes first by number of occurrences and tiebreak with
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

  uint32_t num_buckets = _mach_label_block->index()->numBuckets();
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
  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
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

  auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                num_random_hashes);

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
    // TODO(Geordie / Nick) Right now, we only load MAX_BALANCING_SAMPLES
    // samples to avoid the overhead of loading the entire dataset. It's
    // possible this won't load enough samples to cover all classes.
    // We may try to keep streaming data until all classes are covered or load
    // the entire dataset and see if it makes a difference.
    auto optional_samples =
        _dataset_factory->getLabeledDatasetLoader(data, /* shuffle= */ true)
            ->loadSome(
                /* batch_size= */ defaults::MAX_BALANCING_SAMPLES_TO_LOAD,
                /* num_batches= */ 1, /* verbose= */ false);

    if (!optional_samples) {
      throw std::invalid_argument("No data found for training.");
    }

    auto samples = *optional_samples;
    for (uint32_t i = 0; i < samples.front()->len(); i++) {
      const BoltVector& doc_id_vec = samples.at(2)->at(0)[i];
      if (doc_id_vec.len < 1) {
        continue;
      }

      uint32_t doc_id = doc_id_vec.active_neurons[0];

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

  std::vector<std::pair<bolt::TensorList, bolt::TensorList>> batches;

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

    auto input_tensor =
        bolt::Tensor::convert(BoltBatch(std::move(inputs)), input_dim);

    auto label_tensor =
        bolt::Tensor::convert(BoltBatch(std::move(labels)), label_dim);

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
  warnOnNonHashBasedMetrics(metrics);

  auto dataset = _dataset_factory->getLabeledDatasetLoader(balancing_data,
                                                           /* shuffle= */ true);

  auto associate_samples = getAssociateSamples(source_target_samples);

  std::mt19937 rng(global_random::nextSeed());

  for (auto& [input, hashes] : associate_samples) {
    BoltVector input_vec =
        _dataset_factory->featurizeInput(input).at(0)->getVector(0);
    BoltVector doc_id = BoltVector::singleElementSparseVector(0);
    for (uint32_t i = 0; i < n_association_samples; i++) {
      dataset->manuallyAddToBuffer(
          {input_vec, makeLabelFromHashes(hashes, n_buckets, rng), doc_id});
    }
  }

  return _classifier->train(dataset, learning_rate, epochs,
                            getMetrics(metrics, "train_"),
                            /* val_dataset */ nullptr, /* val_metrics= */ {},
                            /* callbacks= */ {}, options, /*comm= */ nullptr);
}

py::object UDTMachClassifier::associateColdStart(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto metadata = getColdStartMetaData();

  auto cold_start_balancing_data = cold_start::preprocessColdStartTrainSource(
      balancing_data, strong_column_names, weak_column_names, _dataset_factory,
      metadata, /* variable_length= */ std::nullopt);

  return associateTrain(cold_start_balancing_data, source_target_samples,
                        n_buckets, n_association_samples, learning_rate, epochs,
                        metrics, options);
}

void UDTMachClassifier::setDecodeParams(uint32_t top_k_to_return,
                                        uint32_t num_buckets_to_eval) {
  if (top_k_to_return == 0 || num_buckets_to_eval == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  uint32_t num_buckets = _mach_label_block->index()->numBuckets();
  if (num_buckets_to_eval > num_buckets) {
    throw std::invalid_argument(
        "Cannot eval with num_buckets_to_eval greater than " +
        std::to_string(num_buckets) + ".");
  }

  uint32_t num_classes = _mach_label_block->index()->numEntities();
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
  // block allows indexes with different number of hashes but not output ranges
  _mach_label_block->setIndex(index);

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

  bolt::ComputationPtr output = model->outputs().front();
  bolt::ComputationPtr hash_labels = model->labels().front();
  bolt::ComputationPtr true_class_labels = model->labels().back();
  bolt::LossPtr loss = model->losses().front();

  InputMetrics metrics;
  for (const auto& name : metric_names) {
    if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 10, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachPrecision>(
          _mach_label_block->index(), _num_buckets_to_eval, output,
          true_class_labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachRecall>(
          _mach_label_block->index(), _num_buckets_to_eval, output,
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
  archive(cereal::base_class<UDTBackend>(this), _classifier, _mach_label_block,
          _dataset_factory, _pre_hashed_labels_dataset_factory,
          _default_top_k_to_return, _num_buckets_to_eval,
          _mach_sampling_threshold, _rlhf_sampler);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)