#include "UDTMach.h"
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/callbacks/LambdaOnStoppedCallback.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/MachPrecision.h>
#include <bolt/src/train/metrics/MachRecall.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <dataset/src/DataSource.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/Version.h>
#include <utils/text/StringManipulation.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <random>
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

inline uint32_t expectInteger(const Label& label) {
  if (!std::holds_alternative<uint32_t>(label)) {
    throw std::invalid_argument("Must use integer label.");
  }
  return std::get<uint32_t>(label);
}

UDTMach::UDTMach(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const CategoricalDataTypePtr& target_config,
    const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    config::ArgumentMap user_args)
    : _default_top_k_to_return(defaults::MACH_TOP_K_TO_RETURN),
      _num_buckets_to_eval(defaults::MACH_NUM_BUCKETS_TO_EVAL) {
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
      "extreme_output_dim", "integer",
      autotuneMachOutputDim(target_config->expectNClasses()));
  uint32_t num_hashes = user_args.get<uint32_t>(
      "extreme_num_hashes", "integer",
      autotuneMachNumHashes(target_config->expectNClasses(), num_buckets));

  uint32_t softmax = user_args.get<bool>("softmax", "bool", false);

  _classifier = utils::Classifier::make(
      utils::buildModel(
          /* input_dim= */ input_dim, /* output_dim= */ num_buckets,
          /* args= */ user_args, /* model_config= */ model_config,
          /* use_sigmoid_bce = */ !softmax, /* mach= */ true),
      user_args.get<bool>("freeze_hash_tables", "boolean",
                          defaults::FREEZE_HASH_TABLES));

  if (!target_config->isInteger()) {
    throw std::invalid_argument(
        "Option type='int' for the target data type must be specified when "
        "using extreme "
        "classification options.");
  }

  dataset::mach::MachIndexPtr mach_index = dataset::mach::MachIndex::make(
      /* num_buckets = */ num_buckets, /* num_hashes = */ num_hashes);

  if (user_args.contains("mach_index_seed")) {
    mach_index->setSeed(user_args.get<uint32_t>("mach_index_seed", "integer"));
  }

  auto temporal_relationships = TemporalRelationshipsAutotuner::autotune(
      input_data_types, temporal_tracking_relationships,
      tabular_options.lookahead);

  data::ValueFillType value_fill =
      softmax ? data::ValueFillType::SumToOne : data::ValueFillType::Ones;

  _featurizer = std::make_shared<MachFeaturizer>(
      input_data_types, temporal_relationships, target_name, mach_index,
      tabular_options, value_fill);

  _mach_sampling_threshold = user_args.get<float>(
      "mach_sampling_threshold", "float", defaults::MACH_SAMPLING_THRESHOLD);

  updateSamplingStrategy();

  if (user_args.get<bool>("rlhf", "bool", false)) {
    size_t num_balancing_docs = user_args.get<uint32_t>(
        "rlhf_balancing_docs", "int", defaults::MAX_BALANCING_DOCS);
    size_t num_balancing_samples_per_doc =
        user_args.get<uint32_t>("rlhf_balancing_samples_per_doc", "int",
                                defaults::MAX_BALANCING_SAMPLES_PER_DOC);

    enableRlhf(num_balancing_docs, num_balancing_samples_per_doc);
  }

  std::cout
      << "Initialized a UniversalDeepTransformer for Extreme Classification"
      << std::endl;
}

UDTMach::UDTMach(const MachInfo& mach_info)
    : _classifier(mach_info.classifier),
      _default_top_k_to_return(mach_info.default_top_k_to_return),
      _num_buckets_to_eval(mach_info.num_buckets_to_eval),
      _mach_sampling_threshold(mach_info.mach_sampling_threshold) {
  auto text_transform = std::make_shared<data::TextCompat>(
      mach_info.text_column_name, FEATURIZED_INDICES, FEATURIZED_VALUES,
      mach_info.text_block->tokenizer(), mach_info.text_block->encoder(),
      mach_info.text_block->lowercase(), mach_info.text_block->featureDim(),
      mach_info.feature_hash_range);

  _featurizer = std::make_shared<MachFeaturizer>(
      text_transform,
      data::OutputColumnsList{
          data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)},
      mach_info.label_column_name, mach_info.mach_index,
      mach_info.csv_delimiter, mach_info.label_delimiter);

  if (mach_info.balancing_samples) {
    _balancing_samples = BalancingSamples(
        /*indices_col=*/FEATURIZED_INDICES, /*values_col=*/FEATURIZED_VALUES,
        /*labels_col=*/MACH_LABELS, /*doc_ids_col=*/MACH_DOC_IDS,
        /*indices_dim=*/model()->inputDims().at(0),
        /*label_dim=*/getIndex()->numBuckets(),
        /*sampler=*/mach_info.balancing_samples.value());
  }
}

py::object UDTMach::train(const dataset::DataSourcePtr& data,
                          float learning_rate, uint32_t epochs,
                          const std::vector<std::string>& train_metrics,
                          const dataset::DataSourcePtr& val_data,
                          const std::vector<std::string>& val_metrics,
                          const std::vector<CallbackPtr>& callbacks,
                          TrainOptions options,
                          const bolt::DistributedCommPtr& comm,
                          py::kwargs kwargs) {
  insertNewDocIds(data);

  addBalancingSamples(data);

  auto splade_config = getSpladeConfig(kwargs);
  bool splade_in_val = getSpladeValidationOption(kwargs);

  auto train_data_loader = _featurizer->getDataLoader(
      data, options.batchSize(), /* shuffle= */ true, options.verbose,
      splade_config, options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader = _featurizer->getDataLoader(
        val_data, defaults::BATCH_SIZE, /* shuffle= */ false, options.verbose,
        splade_in_val ? splade_config : std::nullopt);
  }

  return _classifier->train(train_data_loader, learning_rate, epochs,
                            getMetrics(train_metrics, "train_"),
                            val_data_loader, getMetrics(val_metrics, "val_"),
                            callbacks, options, comm);
}

py::object UDTMach::trainBatch(const MapInputBatch& batch,
                               float learning_rate) {
  auto& model = _classifier->model();

  _featurizer->insertNewDocIds(data::ColumnMap::fromMapInputBatch(batch));
  updateSamplingStrategy();

  auto [inputs, labels] = _featurizer->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  return py::none();
}

py::object UDTMach::trainWithHashes(const MapInputBatch& batch,
                                    float learning_rate) {
  auto& model = _classifier->model();

  auto [inputs, labels] = _featurizer->featurizeTrainWithHashesBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  return py::none();
}

py::object UDTMach::evaluate(const dataset::DataSourcePtr& data,
                             const std::vector<std::string>& metrics,
                             bool sparse_inference, bool verbose,
                             py::kwargs kwargs) {
  auto splade_config = getSpladeConfig(kwargs);

  auto data_loader =
      _featurizer->getDataLoader(data, defaults::BATCH_SIZE,
                                 /* shuffle= */ false, verbose, splade_config);

  return _classifier->evaluate(data_loader, getMetrics(metrics, "val_"),
                               sparse_inference, verbose);
}

py::object UDTMach::predict(const MapInput& sample, bool sparse_inference,
                            bool return_predicted_class,
                            std::optional<uint32_t> top_k) {
  auto output =
      predictBatch({sample}, sparse_inference, return_predicted_class, top_k);
  return output.cast<py::list>()[0];
}

py::object UDTMach::predictBatch(const MapInputBatch& samples,
                                 bool sparse_inference,
                                 bool return_predicted_class,
                                 std::optional<uint32_t> top_k) {
  return py::cast(predictBatchImpl(samples, sparse_inference,
                                   return_predicted_class, top_k));
}

py::object UDTMach::predictActivationsBatch(const MapInputBatch& samples,
                                            bool sparse_inference) {
  return bolt::python::tensorToNumpy(
      _classifier->model()
          ->forward(_featurizer->featurizeInputBatch(samples), sparse_inference)
          .at(0));
}

std::vector<std::vector<std::pair<uint32_t, double>>> UDTMach::predictBatchImpl(
    const MapInputBatch& samples, bool sparse_inference,
    bool return_predicted_class, std::optional<uint32_t> top_k) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  auto outputs =
      _classifier->model()
          ->forward(_featurizer->featurizeInputBatch(samples), sparse_inference)
          .at(0);

  uint32_t num_classes = getIndex()->numEntities();

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
    auto predictions = getIndex()->decode(
        /* output = */ vector,
        /* top_k = */ k,
        /* num_buckets_to_eval = */ _num_buckets_to_eval);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
}

py::object UDTMach::scoreBatch(const MapInputBatch& samples,
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
                     ->forward(_featurizer->featurizeInputBatch(samples),
                               /* use_sparsity= */ false)
                     .at(0);

  size_t batch_size = samples.size();
  std::vector<std::vector<std::pair<uint32_t, double>>> scores(samples.size());

  const auto& index = getIndex();
#pragma omp parallel for default(none) shared( \
    entities, outputs, scores, top_k, batch_size, index) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs->getVector(i);
    scores[i] = index->scoreEntities(vector, entities[i], top_k);
  }

  return py::cast(scores);
}

py::object UDTMach::predictHashes(const MapInput& sample, bool sparse_inference,
                                  bool force_non_empty,
                                  std::optional<uint32_t> num_hashes) {
  return py::cast(
      predictHashesImpl({sample}, sparse_inference, force_non_empty, num_hashes)
          .at(0));
}

py::object UDTMach::predictHashesBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool force_non_empty,
                                       std::optional<uint32_t> num_hashes) {
  return py::cast(predictHashesImpl(samples, sparse_inference, force_non_empty,
                                    num_hashes));
}

std::vector<std::vector<uint32_t>> UDTMach::predictHashesImpl(
    const MapInputBatch& samples, bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  auto outputs =
      _classifier->model()
          ->forward(_featurizer->featurizeInputBatch(samples), sparse_inference)
          .at(0);

  uint32_t k = num_hashes.value_or(getIndex()->numHashes());

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, k, force_non_empty)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    if (force_non_empty) {
      heap = getIndex()->topKNonEmptyBuckets(output, k);
    } else {
      heap = output.topKNeurons(k);
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

py::object UDTMach::outputCorrectness(const MapInputBatch& samples,
                                      const std::vector<uint32_t>& labels,
                                      bool sparse_inference,
                                      std::optional<uint32_t> num_hashes) {
  std::vector<std::vector<uint32_t>> top_buckets = predictHashesImpl(
      samples, sparse_inference, /* force_non_empty = */ true, num_hashes);

  std::vector<uint32_t> matching_buckets(labels.size());
  std::exception_ptr hashes_err;

#pragma omp parallel for default(none) \
    shared(labels, top_buckets, matching_buckets, hashes_err)
  for (uint32_t i = 0; i < labels.size(); i++) {
    try {
      std::vector<uint32_t> hashes = getIndex()->getHashes(labels[i]);
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

void UDTMach::setModel(const ModelPtr& model) {
  bolt::ModelPtr& curr_model = _classifier->model();

  utils::verifyCanSetModel(curr_model, model);

  curr_model = model;
}

py::object UDTMach::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks_in, TrainOptions options,
    const bolt::DistributedCommPtr& comm, const py::kwargs& kwargs) {
  insertNewDocIds(data);

  addBalancingSamples(data, strong_column_names, weak_column_names,
                      variable_length);

  auto splade_config = getSpladeConfig(kwargs);
  auto splade_in_val = getSpladeValidationOption(kwargs);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader = _featurizer->getDataLoader(
        val_data, defaults::BATCH_SIZE,
        /* shuffle= */ false, options.verbose,
        /*splade_config=*/splade_in_val ? splade_config : std::nullopt);
  }

  bool stopped = false;

  auto callbacks = callbacks_in;
  callbacks.push_back(
      std::make_shared<bolt::callbacks::LambdaOnStoppedCallback>(
          bolt::callbacks::LambdaOnStoppedCallback(
              [&stopped]() { stopped = true; })));

  // TODO(Nicholas): make it so that the spade augmentation is only run once
  // rather than for every epoch if variable length cold start is used.
  uint32_t epoch_step = variable_length.has_value() ? 1 : epochs;

  py::object history;
  for (uint32_t e = 0; e < epochs; e += epoch_step) {
    auto train_data_loader = _featurizer->getColdStartDataLoader(
        data, strong_column_names, weak_column_names,
        /* variable_length= */ variable_length,
        /*splade_config=*/splade_config, /* fast_approximation= */ false,
        options.batchSize(), /* shuffle= */ true, options.verbose,
        options.shuffle_config);

    history = _classifier->train(
        train_data_loader, learning_rate, epoch_step,
        getMetrics(train_metrics, "train_"), val_data_loader,
        getMetrics(val_metrics, "val_"), callbacks, options, comm);

    data->restart();
    if (val_data_loader) {
      val_data_loader->restart();
    }

    if (stopped) {
      break;
    }
  }

  return history;
}

py::object UDTMach::embedding(const MapInputBatch& sample) {
  return _classifier->embedding(_featurizer->featurizeInputBatch(sample));
}

py::object UDTMach::entityEmbedding(const Label& label) {
  std::vector<uint32_t> hashed_neurons =
      getIndex()->getHashes(expectInteger(label));

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

void UDTMach::updateSamplingStrategy() {
  auto mach_index = getIndex();

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

void UDTMach::insertNewDocIds(const dataset::DataSourcePtr& data) {
  _featurizer->insertNewDocIds(data);
  updateSamplingStrategy();
}

void UDTMach::introduceDocuments(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool load_balancing, bool fast_approximation,
    bool verbose, bool sort_random_hashes) {
  (void)verbose;
  // TODO(Nicholas): add progress bar here.

  const auto& mach_index = getIndex();

  auto data_and_doc_ids = _featurizer->featurizeForIntroduceDocuments(
      data, strong_column_names, weak_column_names, fast_approximation,
      defaults::BATCH_SIZE);

  uint32_t num_buckets_to_sample =
      load_balancing
          ? mach_index->numBuckets()
          : num_buckets_to_sample_opt.value_or(mach_index->numHashes());

  std::unordered_map<uint32_t, std::vector<std::vector<ValueIndexPair>>>
      top_k_per_doc;

  bolt::python::CtrlCCheck ctrl_c_check;

  for (const auto& [input, doc_ids] : data_and_doc_ids) {
    // Note: using sparse inference here could cause issues because the
    // mach index sampler will only return nonempty buckets, which could
    // cause new docs to only be mapped to buckets already containing
    // entities.
    auto scores = _classifier->model()->forward(input).at(0);

    for (uint32_t i = 0; i < scores->batchSize(); i++) {
      uint32_t label = doc_ids[i];
      if (load_balancing) {
        top_k_per_doc[label].push_back(scores->getVector(i).valueIndexPairs());
      } else {
        top_k_per_doc[label].push_back(priorityQueueToVector(
            scores->getVector(i).topKNeurons(num_buckets_to_sample)));
      }
    }

    ctrl_c_check();
  }

  uint32_t approx_num_hashes_per_bucket =
      mach_index->approxNumHashesPerBucket(top_k_per_doc.size());

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(
        std::move(top_ks), num_buckets_to_sample, approx_num_hashes_per_bucket,
        num_random_hashes, load_balancing, sort_random_hashes);
    mach_index->insert(doc, hashes);

    ctrl_c_check();
  }

  addBalancingSamples(data, strong_column_names, weak_column_names,
                      /*variable_length=*/std::nullopt);

  updateSamplingStrategy();
}

void UDTMach::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes,
    bool load_balancing, bool sort_random_hashes) {
  auto samples = _featurizer->featurizeInputColdStart(
      document, strong_column_names, weak_column_names);

  introduceLabelHelper(samples, new_label, num_buckets_to_sample,
                       num_random_hashes, load_balancing, sort_random_hashes);
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

std::vector<uint32_t> UDTMach::topHashesForDoc(
    std::vector<std::vector<ValueIndexPair>>&& top_k_per_sample,
    uint32_t num_buckets_to_sample, uint32_t approx_num_hashes_per_bucket,
    uint32_t num_random_hashes, bool load_balancing,
    bool sort_random_hashes) const {
  const auto& mach_index = getIndex();

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
    for (const auto& [activation, active_neuron] : top_k) {
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, activation};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += activation;
      }
    }
  }

  uint32_t num_buckets = mach_index->numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, num_buckets - 1);
  std::mt19937 rand(global_random::nextSeed());

  if (sort_random_hashes) {
    for (uint32_t i = 0; i < num_random_hashes; i++) {
      uint32_t active_neuron = int_dist(rand);
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, 0};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += 0;
      }
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
              [&mach_index, &cmp, approx_num_hashes_per_bucket, load_balancing](
                  const auto& lhs, const auto& rhs) {
                size_t lhs_size = mach_index->bucketSize(lhs.first);
                size_t rhs_size = mach_index->bucketSize(rhs.first);

                // Give preference to emptier buckets. If buckets are
                // equally empty, use one with the best score.

                if (load_balancing) {
                  if (lhs_size < approx_num_hashes_per_bucket &&
                      rhs_size < approx_num_hashes_per_bucket) {
                    return cmp(lhs, rhs);
                  }
                }
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

  uint32_t num_informed_hashes =
      sort_random_hashes ? num_hashes : (num_hashes - num_random_hashes);

  for (uint32_t i = 0; i < num_informed_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes.push_back(hash);
  }

  if (!sort_random_hashes) {
    for (uint32_t i = 0; i < num_random_hashes; i++) {
      if (load_balancing) {
        uint32_t random_hash;

        do {
          random_hash = int_dist(rand);
        } while (mach_index->bucketSize(random_hash) >=
                 approx_num_hashes_per_bucket);

        new_hashes.push_back(random_hash);

      } else {
        new_hashes.push_back(int_dist(rand));
      }
    }
  }

  return new_hashes;
}

void UDTMach::introduceLabel(const MapInputBatch& samples,
                             const Label& new_label,
                             std::optional<uint32_t> num_buckets_to_sample_opt,
                             uint32_t num_random_hashes, bool load_balancing,
                             bool sort_random_hashes) {
  introduceLabelHelper(_featurizer->featurizeInputBatch(samples), new_label,
                       num_buckets_to_sample_opt, num_random_hashes,
                       load_balancing, sort_random_hashes);
}

void UDTMach::introduceLabelHelper(
    const bolt::TensorList& samples, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool load_balancing, bool sort_random_hashes) {
  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
  auto output =
      _classifier->model()->forward(samples, /* use_sparsity = */ false).at(0);

  uint32_t num_buckets_to_sample =
      load_balancing
          ? getIndex()->numBuckets()
          : num_buckets_to_sample_opt.value_or(getIndex()->numHashes());

  const auto& mach_index = getIndex();

  std::vector<std::vector<ValueIndexPair>> top_ks;
  for (uint32_t i = 0; i < output->batchSize(); i++) {
    if (load_balancing) {
      top_ks.push_back(output->getVector(i).valueIndexPairs());
    } else {
      top_ks.push_back(priorityQueueToVector(
          output->getVector(i).topKNeurons(num_buckets_to_sample)));
    }
  }

  uint32_t approx_num_hashes_per_bucket =
      mach_index->approxNumHashesPerBucket(samples.size());

  auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                approx_num_hashes_per_bucket, num_random_hashes,
                                load_balancing, sort_random_hashes);

  getIndex()->insert(expectInteger(new_label), hashes);

  updateSamplingStrategy();
}

void UDTMach::forget(const Label& label) {
  getIndex()->erase(expectInteger(label));

  if (getIndex()->numEntities() == 0) {
    std::cout << "Warning. Every learned class has been forgotten. The model "
                 "will currently return nothing on calls to evaluate, "
                 "predict, or predictBatch."
              << std::endl;
  }

  updateSamplingStrategy();
}

void UDTMach::addBalancingSamples(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length) {
  if (_balancing_samples) {
    data->restart();

    // TODO(Geordie / Nick) Right now, we only load MAX_BALANCING_SAMPLES
    // samples to avoid the overhead of loading the entire dataset. It's
    // possible this won't load enough samples to cover all classes.
    // We may try to keep streaming data until all classes are covered or load
    // the entire dataset and see if it makes a difference. For now we just
    // sample from 5x more rows than we need samples, to hopefully get a wider
    // range of samples.
    auto samples = _featurizer->getBalancingSamples(
        data, strong_column_names, weak_column_names, variable_length,
        /*n_balancing_samples=*/defaults::MAX_BALANCING_SAMPLES_TO_LOAD);

    _balancing_samples->addSamples(samples);

    data->restart();
  }
}

void UDTMach::requireRLHFSampler() {
  if (!_balancing_samples) {
    throw std::runtime_error(
        "This model was not configured to support rlhf. Please pass "
        "{'rlhf': "
        "True} in the model options or call enable_rlhf().");
  }
}

void UDTMach::enableRlhf(uint32_t num_balancing_docs,
                         uint32_t num_balancing_samples_per_doc) {
  if (_balancing_samples.has_value()) {
    return;
  }

  _balancing_samples = BalancingSamples(
      /*indices_col=*/FEATURIZED_INDICES, /*values_col=*/FEATURIZED_VALUES,
      /*labels_col=*/MACH_LABELS, /*doc_ids_col=*/MACH_DOC_IDS,
      /*indices_dim=*/model()->inputDims().at(0),
      /*label_dim=*/getIndex()->numBuckets(),
      /*max_docs=*/num_balancing_docs,
      /*max_samples_per_doc=*/num_balancing_samples_per_doc);
}

void UDTMach::associate(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs,
    bool force_non_empty, size_t batch_size) {
  auto teaching_samples = getAssociateSamples(
      rlhf_samples, n_buckets, n_association_samples, force_non_empty);

  teach(teaching_samples, rlhf_samples.size() * n_balancing_samples,
        learning_rate, epochs, batch_size);
}

void UDTMach::upvote(
    const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
    uint32_t n_upvote_samples, uint32_t n_balancing_samples,
    float learning_rate, uint32_t epochs, size_t batch_size) {
  std::vector<RlhfSample> teaching_samples;

  const auto& mach_index = getIndex();

  teaching_samples.reserve(rlhf_samples.size() * n_upvote_samples);
  for (const auto& [source, target] : rlhf_samples) {
    RlhfSample sample = {source, mach_index->getHashes(target)};

    for (size_t i = 0; i < n_upvote_samples; i++) {
      teaching_samples.push_back(sample);
    }
  }

  teach(teaching_samples, rlhf_samples.size() * n_balancing_samples,
        learning_rate, epochs, batch_size);
}

void UDTMach::teach(const std::vector<RlhfSample>& rlhf_samples,
                    uint32_t n_balancing_samples, float learning_rate,
                    uint32_t epochs, size_t batch_size) {
  requireRLHFSampler();

  auto balancing_samples =
      _balancing_samples->balancingSamples(n_balancing_samples);

  auto columns = _featurizer->featurizeRlhfSamples(rlhf_samples);
  columns = columns.concat(balancing_samples);
  columns.shuffle();

  auto [data, labels] = _featurizer->columnsToTensors(columns, batch_size);

  for (uint32_t e = 0; e < epochs; e++) {
    for (size_t i = 0; i < data.size(); i++) {
      _classifier->model()->trainOnBatch(data.at(i), labels.at(i));
      _classifier->model()->updateParameters(learning_rate);
    }
  }
}

std::vector<RlhfSample> UDTMach::getAssociateSamples(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    size_t n_buckets, size_t n_association_samples, bool force_non_empty) {
  std::string text_column = _featurizer->textDatasetConfig().textColumn();
  MapInputBatch batch;
  for (const auto& [_, target] : rlhf_samples) {
    batch.push_back({{text_column, target}});
  }

  auto all_predicted_hashes =
      predictHashesImpl(batch, /* sparse_inference = */ false,
                        /* force_non_empty = */ force_non_empty);

  std::mt19937 rng(global_random::nextSeed());

  std::vector<RlhfSample> associate_samples;
  associate_samples.reserve(rlhf_samples.size() * n_association_samples);

  for (size_t i = 0; i < rlhf_samples.size(); i++) {
    for (size_t j = 0; j < n_association_samples; j++) {
      const std::vector<uint32_t>& all_buckets = all_predicted_hashes[i];
      std::vector<uint32_t> sampled_buckets;
      std::sample(all_buckets.begin(), all_buckets.end(),
                  std::back_inserter(sampled_buckets), n_buckets, rng);

      associate_samples.emplace_back(rlhf_samples[i].first, sampled_buckets);
    }
  }

  return associate_samples;
}

py::object UDTMach::associateTrain(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  return associateColdStart(balancing_data, {}, {}, rlhf_samples, n_buckets,
                            n_association_samples, learning_rate, epochs,
                            metrics, options);
}

py::object UDTMach::associateColdStart(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  insertNewDocIds(balancing_data);

  warnOnNonHashBasedMetrics(metrics);

  if (options.max_in_memory_batches) {
    throw std::invalid_argument(
        "Streaming is not supported for associate_train/associate_cold_start. "
        "Please pass max_in_memory_batches=None.");
  }

  auto featurized_data = _featurizer->featurizeDataset(
      balancing_data, strong_column_names, weak_column_names,
      /*variable_length=*/std::nullopt);

  auto associate_samples =
      getAssociateSamples(rlhf_samples, n_buckets, n_association_samples);

  auto featurized_rlhf_data =
      _featurizer->featurizeRlhfSamples(associate_samples);

  auto columns = featurized_data.concat(featurized_rlhf_data);
  columns.shuffle();

  auto dataset = _featurizer->columnsToTensors(columns, options.batchSize());

  bolt::Trainer trainer(_classifier->model());

  auto output_metrics =
      trainer.train(/* train_data= */ dataset,
                    /* learning_rate= */ learning_rate, /* epochs= */ epochs,
                    /* train_metrics= */ getMetrics(metrics, "train_"),
                    /* validation_data= */ {},
                    /* validation_metrics= */ {},
                    /* steps_per_validation= */ {},
                    /* use_sparsity_in_validation= */ false,
                    /* callbacks= */ {},
                    /* autotune_rehash_rebuild= */ true,
                    /* verbose= */ options.verbose,
                    /* logging_interval= */ options.logging_interval);

  return py::cast(output_metrics);
}

py::object UDTMach::coldStartWithBalancingSamples(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& train_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const std::optional<data::VariableLengthConfig>& variable_length) {
  insertNewDocIds(data);
  requireRLHFSampler();

  addBalancingSamples(data, strong_column_names, weak_column_names,
                      variable_length);

  warnOnNonHashBasedMetrics(train_metrics);

  if (options.max_in_memory_batches) {
    throw std::invalid_argument(
        "Streaming is not supported for cold_start_with_balancing_samples. "
        "Please pass max_in_memory_batches=None.");
  }

  auto featurized_data = _featurizer->featurizeDataset(
      data, strong_column_names, weak_column_names, variable_length);

  data::ColumnMap balancing_data({});
  if (featurized_data.numRows() < _balancing_samples->totalBalancingSamples()) {
    balancing_data =
        _balancing_samples->balancingSamples(featurized_data.numRows());
  } else {
    balancing_data = _balancing_samples->allBalancingSamples();
  }

  auto columns = featurized_data.concat(balancing_data);
  columns.shuffle();

  auto dataset = _featurizer->columnsToTensors(columns, options.batchSize());

  bolt::Trainer trainer(_classifier->model());

  auto output_metrics =
      trainer.train(/* train_data= */ dataset,
                    /* learning_rate= */ learning_rate, /* epochs= */ epochs,
                    /* train_metrics= */ getMetrics(train_metrics, "train_"),
                    /* validation_data= */ {},
                    /* validation_metrics= */ {},
                    /* steps_per_validation= */ {},
                    /* use_sparsity_in_validation= */ false,
                    /* callbacks= */ callbacks,
                    /* autotune_rehash_rebuild= */ true,
                    /* verbose= */ options.verbose,
                    /* logging_interval= */ options.logging_interval);

  return py::cast(output_metrics);
}

void UDTMach::setDecodeParams(uint32_t top_k_to_return,
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

void UDTMach::setIndex(const dataset::mach::MachIndexPtr& index) {
  // block allows indexes with different number of hashes but not output ranges
  _featurizer->state()->setMachIndex(index);

  updateSamplingStrategy();
}

void UDTMach::setMachSamplingThreshold(float threshold) {
  _mach_sampling_threshold = threshold;
  updateSamplingStrategy();
}

InputMetrics UDTMach::getMetrics(const std::vector<std::string>& metric_names,
                                 const std::string& prefix) {
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

void UDTMach::warnOnNonHashBasedMetrics(
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

bolt::TensorPtr UDTMach::placeholderDocIds(uint32_t batch_size) {
  return bolt::Tensor::sparse(batch_size, std::numeric_limits<uint32_t>::max(),
                              /* nonzeros= */ 1);
}

ar::ConstArchivePtr UDTMach::toArchive(bool with_optimizer) const {
  auto map = _classifier->toArchive(with_optimizer);
  map->set("type", ar::str(type()));
  map->set("featurizer", _featurizer->toArchive());

  map->set("default_top_k_to_return", ar::u64(_default_top_k_to_return));
  map->set("num_buckets_to_eval", ar::u64(_num_buckets_to_eval));
  map->set("mach_sampling_threshold", ar::f32(_mach_sampling_threshold));

  if (_balancing_samples) {
    map->set("balancing_samples", _balancing_samples->toArchive());
  }

  return map;
}

std::unique_ptr<UDTMach> UDTMach::fromArchive(const ar::Archive& archive) {
  return std::make_unique<UDTMach>(archive);
}

UDTMach::UDTMach(const ar::Archive& archive)
    : _classifier(utils::Classifier::fromArchive(archive)),
      _featurizer(MachFeaturizer::fromArchive(*archive.get("featurizer"))),
      _default_top_k_to_return(archive.u64("default_top_k_to_return")),
      _num_buckets_to_eval(archive.u64("num_buckets_to_eval")),
      _mach_sampling_threshold(
          archive.getAs<ar::F32>("mach_sampling_threshold")) {
  if (archive.contains("balancing_samples")) {
    _balancing_samples = BalancingSamples(*archive.get("balancing_samples"));
  }
}

template void UDTMach::serialize(cereal::BinaryInputArchive&,
                                 const uint32_t version);
template void UDTMach::serialize(cereal::BinaryOutputArchive&,
                                 const uint32_t version);

template <class Archive>
void UDTMach::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_MACH_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_MACH_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_MACH_CLASSIFIER_VERSION after
  // serialization changes
  archive(cereal::base_class<UDTBackend>(this), _classifier, _featurizer,
          _default_top_k_to_return, _num_buckets_to_eval,
          _mach_sampling_threshold, _balancing_samples);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMach)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMach,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)