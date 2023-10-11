#include "UDTMachClassifier.h"
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/python_bindings/NumpyConversions.h>
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
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/featurization/UDTTransformationFactory.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Mach.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
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
#include <string>
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

namespace feat = thirdai::data;

config::ArgumentMap setNeuraldbDefaultArgs(config::ArgumentMap&& user_args) {
  user_args.insert<uint32_t>("embedding_dimension", 2048);
  user_args.insert<uint32_t>("extreme_output_dim", 50000);
  user_args.insert<uint32_t>("extreme_num_hashes", 8);
  user_args.insert<bool>("use_bias", false);
  user_args.insert<bool>("use_tanh", true);
  user_args.insert<bool>("rlhf", true);
  return user_args;
}

UDTMachClassifier::UDTMachClassifier(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target,
    const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    config::ArgumentMap user_args)
    : _delimiter(tabular_options.delimiter),
      _default_top_k_to_return(defaults::MACH_TOP_K_TO_RETURN),
      _num_buckets_to_eval(defaults::MACH_NUM_BUCKETS_TO_EVAL) {
  (void)target_config;

  if (!integer_target) {
    throw std::invalid_argument(
        "Option 'integer_target=True' must be specified when using extreme "
        "classification options.");
  }

  uint32_t input_dim = tabular_options.feature_hash_range;

  if (user_args.get<bool>("neural_db", "boolean", /* default_val= */ false)) {
    input_dim = 50000;
    user_args = setNeuraldbDefaultArgs(std::move(user_args));
  }

  uint32_t num_buckets = user_args.get<uint32_t>(
      "extreme_output_dim", "integer", autotuneMachOutputDim(n_target_classes));
  uint32_t num_hashes = user_args.get<uint32_t>(
      "extreme_num_hashes", "integer",
      autotuneMachNumHashes(n_target_classes, num_buckets));
  bool freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                                defaults::FREEZE_HASH_TABLES);
  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart?
  float mach_sampling_threshold = user_args.get<float>(
      "mach_sampling_threshold", "float", defaults::MACH_SAMPLING_THRESHOLD);

  std::optional<RLHFSampler> rlhf_sampler;
  if (user_args.get<bool>("rlhf", "bool", false)) {
    size_t num_balancing_docs = user_args.get<uint32_t>(
        "rlhf_balancing_docs", "int", defaults::MAX_BALANCING_DOCS);
    size_t num_balancing_samples_per_doc =
        user_args.get<uint32_t>("rlhf_balancing_samples_per_doc", "int",
                                defaults::MAX_BALANCING_SAMPLES_PER_DOC);

    enableRlhf(num_balancing_docs, num_balancing_samples_per_doc);
  }

  _mach = utils::Mach::make(
      /* input_dim= */ input_dim,
      /* num_buckets= */ num_buckets,
      /* num_hashes= */ num_hashes,
      /* mach_sampling_threshold= */ mach_sampling_threshold,
      /* freeze_hash_tables= */ freeze_hash_tables,
      /* args= */ user_args,
      /* model_config= */ model_config);

  _mach->randomlyAssignBuckets(n_target_classes);

  _state = feat::State::make();

  _featurizer = UDTTransformationFactory::make(
      /* data_types= */ input_data_types,
      /* user_temporal_relationships= */ temporal_tracking_relationships,
      /* label_column= */ target_name,
      /* label_value_fill= */ feat::ValueFillType::Ones,
      /* options= */ tabular_options);
}

py::object UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  auto train_csv_iter = feat::CsvIterator::make(data, _delimiter);
  auto train_iter = feat::Chain::make(std::move(train_csv_iter), _state)
                        ->then(_featurizer->trainInputTransform())
                        ->then(_featurizer->labelTransform());

  if (_state->rlhfSampler()) {
    train_iter = train_iter->then(_featurizer->storeBalancers());
  }

  std::shared_ptr<feat::Chain> valid_iter;
  if (val_data) {
    auto valid_csv_iter = feat::CsvIterator::make(data, _delimiter);
    valid_iter = feat::Chain::make(std::move(valid_csv_iter), _state)
                     ->then(_featurizer->trainInputTransform())
                     ->then(_featurizer->labelTransform());
  }

  return py::cast(
      _mach->train(std::move(train_iter), std::move(valid_iter), inputColumns(),
                   docIdColumn(), learning_rate, epochs,
                   /* train_metrics= */ getMetrics(train_metrics, "train_"),
                   /* val_metrics= */ getMetrics(val_metrics, "val_"),
                   /* callbacks= */ callbacks, /* options= */ options,
                   /* comm= */ comm));
}

py::object UDTMachClassifier::trainBatch(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto table = feat::ColumnMap::fromMapInputBatch(batch);
  table = _featurizer->trainInputTransform()->apply(std::move(table), *_state);
  _mach->trainBatch(std::move(table), inputColumns(), docIdColumn(),
                    learning_rate);
  (void)metrics;
  return py::none();
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose,
                                       std::optional<uint32_t> top_k) {
  (void)top_k;
  auto eval_csv_iter = feat::CsvIterator::make(data, _delimiter);
  auto eval_iter = feat::Chain::make(std::move(eval_csv_iter), _state)
                       ->then(_featurizer->trainInputTransform())
                       ->then(_featurizer->labelTransform());

  return py::cast(_mach->evaluate(std::move(eval_iter), inputColumns(),
                                  docIdColumn(),
                                  /* metrics= */ getMetrics(metrics, "val_"),
                                  /* sparse_inference= */ sparse_inference,
                                  /* verbose= */ verbose));
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

  auto table = feat::ColumnMap::fromMapInput(sample);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);

  return py::cast(
      _mach
          ->predict(table, inputColumns(), sparse_inference,
                    /* top_k= */ top_k.value_or(_default_top_k_to_return),
                    /* num_scanned_buckets= */ _num_buckets_to_eval)
          .at(0));
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

  auto table = feat::ColumnMap::fromMapInputBatch(samples);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);

  return py::cast(
      _mach->predict(table, inputColumns(), sparse_inference,
                     /* top_k= */ top_k.value_or(_default_top_k_to_return),
                     /* num_scanned_buckets= */ _num_buckets_to_eval));
}

py::object UDTMachClassifier::outputCorrectness(
    const MapInputBatch& samples, const std::vector<uint32_t>& labels,
    bool sparse_inference, std::optional<uint32_t> num_hashes) {
  auto table = feat::ColumnMap::fromMapInputBatch(samples);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);

  return py::cast(_mach->outputCorrectness(table, inputColumns(), labels,
                                           num_hashes, sparse_inference));
}

void UDTMachClassifier::setModel(const ModelPtr& model) {
  bolt::ModelPtr& curr_model = _mach->model();

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
  auto train_csv_iter = feat::CsvIterator::make(data, _delimiter);
  auto train_iter = feat::Chain::make(train_csv_iter, _state)
                        ->then(_featurizer->unsupAugmenter(
                            /* strong_column_names= */ strong_column_names,
                            /* weak_column_names= */ weak_column_names,
                            /* fast_approximation= */ false))
                        ->then(_featurizer->trainInputTransform())
                        ->then(_featurizer->labelTransform());

  if (_state->rlhfSampler()) {
    train_iter = train_iter->then(_featurizer->storeBalancers());
  }

  std::shared_ptr<feat::Chain> valid_iter;
  if (val_data) {
    auto valid_csv_iter = feat::CsvIterator::make(data, _delimiter);
    valid_iter = feat::Chain::make(std::move(valid_csv_iter), _state)
                     ->then(_featurizer->trainInputTransform())
                     ->then(_featurizer->labelTransform());
  }

  return py::cast(
      _mach->train(std::move(train_iter), std::move(valid_iter),
                   /* input_columns= */ inputColumns(),
                   /* doc_id_column= */ docIdColumn(),
                   /* learning_rate= */ learning_rate, /* epochs= */ epochs,
                   /* train_metrics= */ getMetrics(train_metrics, "train_"),
                   /* val_metrics= */ getMetrics(val_metrics, "val_"),
                   /* callbacks= */ callbacks, /* options= */ options,
                   /* comm= */ comm));
}

py::object UDTMachClassifier::embedding(const MapInputBatch& sample) {
  auto table = feat::ColumnMap::fromMapInputBatch(sample);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);
  return bolt::python::tensorToNumpy(_mach->embedding(table, inputColumns()));
}

uint32_t expectInteger(const Label& label) {
  if (!std::holds_alternative<uint32_t>(label)) {
    throw std::invalid_argument("Must use integer label.");
  }
  return std::get<uint32_t>(label);
}

py::object UDTMachClassifier::entityEmbedding(const Label& label) {
  auto embedding = _mach->entityEmbedding(expectInteger(label));
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
  // TODO(Nicholas): add progress bar here.

  auto table = feat::CsvIterator::all(data, _delimiter);
  table = _featurizer
              ->unsupAugmenter(strong_column_names, weak_column_names,
                               fast_approximation)
              ->apply(std::move(table), *_state);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);
  table = _featurizer->labelTransform()->apply(std::move(table), *_state);
  table = _featurizer->storeBalancers()->apply(std::move(table), *_state);

  _mach->introduceEntities(
      table, inputColumns(), docIdColumn(),
      /* num_buckets_to_sample_opt= */ num_buckets_to_sample_opt,
      /* num_random_hashes= */ num_random_hashes);
}

feat::ColumnMap addLabelColumn(feat::ColumnMap&& table,
                               const std::string& doc_id_column,
                               uint32_t label) {
  std::vector<uint32_t> labels(table.numRows());
  std::fill(labels.begin(), labels.end(), label);
  table.setColumn(
      /* name= */ doc_id_column,
      /* column= */ feat::ValueColumn<uint32_t>::make(std::move(labels)));
  return table;
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  auto table = feat::ColumnMap::fromMapInput(document);
  table = addLabelColumn(std::move(table), docIdColumn(), expectInteger(label));
  table = _featurizer
              ->unsupAugmenter(strong_column_names, weak_column_names,
                               /* fast_approximation= */ false)
              ->apply(std::move(table), *_state);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);

  _mach->introduceEntities(
      /* table= */ table, inputColumns(), docIdColumn(),
      /* num_buckets_to_sample_opt= */ num_buckets_to_sample_opt,
      /* num_random_hashes= */ num_random_hashes);
}

void UDTMachClassifier::introduceLabel(
    const MapInputBatch& samples, const Label& label,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  auto table = feat::ColumnMap::fromMapInputBatch(samples);
  table = addLabelColumn(std::move(table), docIdColumn(), expectInteger(label));
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);

  _mach->introduceEntities(
      /* table= */ table, inputColumns(), docIdColumn(),
      /* num_buckets_to_sample_opt= */ num_buckets_to_sample_opt,
      /* num_random_hashes= */ num_random_hashes);
}

void UDTMachClassifier::forget(const Label& label) {
  _mach->eraseEntity(expectInteger(label));
}

void UDTMachClassifier::requireRLHFSampler() {
  if (!_state->rlhfSampler()) {
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

auto rlhfBalancerTable(const RLHFSampler& sampler,
                       const std::string& text_column_name,
                       const std::string& id_column_name,
                       feat::Transformation& input_transform,
                       feat::State& state, uint32_t num_samples) {
  auto samples = sampler.balancingSamples(num_samples);
  std::vector<std::string> text_data(samples.size());
  std::vector<uint32_t> id_data(samples.size());
  for (uint32_t i = 0; i < samples.size(); i++) {
    text_data[i] = std::move(samples[i].first);
    id_data[i] = samples[i].second;
  }
  auto text_column = feat::ValueColumn<std::string>::make(std::move(text_data));
  auto id_column = feat::ValueColumn<uint32_t>::make(std::move(id_data));
  auto table = feat::ColumnMap({{text_column_name, std::move(text_column)},
                                {id_column_name, std::move(id_column)}});
  return input_transform.apply(std::move(table), state);
}

auto columnMapFromUpvoteSamples(
    const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples) {
  std::vector<std::string> text_data(rlhf_samples.size());
  std::vector<uint32_t> doc_id_data(rlhf_samples.size());

  for (uint32_t i = 0; i < rlhf_samples.size(); i++) {
    text_data[i] = rlhf_samples[i].first;
    doc_id_data[i] = rlhf_samples[i].second;
  }
}

void UDTMachClassifier::upvote(
    const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
    uint32_t n_upvote_samples, uint32_t n_balancing_samples,
    float learning_rate, uint32_t epochs) {
  auto text_column_name = _featurizer->textDatasetConfig().textColumn();

  std::vector<std::string> text_data(rlhf_samples.size());
  std::vector<uint32_t> id_data(rlhf_samples.size());
  for (uint32_t i = 0; i < rlhf_samples.size(); i++) {
    text_data[i] = rlhf_samples[i].first;
    id_data[i] = rlhf_samples[i].second;
  }
  auto text_column = feat::ValueColumn<std::string>::make(std::move(text_data));
  auto doc_id_column = feat::ValueColumn<uint32_t>::make(std::move(id_data));

  auto upvotes = feat::ColumnMap(
      {{text_column_name, text_column}, {docIdColumn(), doc_id_column}});
  upvotes =
      _featurizer->inferInputTransform()->apply(std::move(upvotes), *_state);
  upvotes = _featurizer->labelTransform()->apply(std::move(upvotes), *_state);

  auto balancers = rlhfBalancerTable(
      /* sampler= */ *_state->rlhfSampler(),
      /* text_column_name= */ _featurizer->textDatasetConfig().textColumn(),
      /* id_column_name= */ docIdColumn(),
      /* input_transform= */ *_featurizer->inferInputTransform(),
      /* state= */ *_state,
      /* num_samples= */ rlhf_samples.size() * n_balancing_samples);

  _mach->upvote(std::move(upvotes), std::move(balancers), inputColumns(),
                docIdColumn(), learning_rate,
                /* repeats= */ n_upvote_samples, /* epochs= */ epochs,
                /* batch_size= */ defaults::ASSOCIATE_BATCH_SIZE);
}

auto columnMapsFromAssociateSamples(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    const TextDatasetConfig& text_dataset) {
  std::vector<std::string> from_data(rlhf_samples.size());
  std::vector<std::string> to_data(rlhf_samples.size());
  for (uint32_t i = 0; i < rlhf_samples.size(); i++) {
    from_data[i] = rlhf_samples[i].first;
    to_data[i] = rlhf_samples[i].second;
  }
  auto from_column = feat::ValueColumn<std::string>::make(std::move(from_data));
  auto to_column = feat::ValueColumn<std::string>::make(std::move(to_data));
  const auto& text_column_name = text_dataset.textColumn();

  return std::make_pair(
      feat::ColumnMap({{text_column_name, std::move(from_column)}}),
      feat::ColumnMap({{text_column_name, std::move(to_column)}}));
}

py::object UDTMachClassifier::associateTrain(
    feat::ColumnMap featurized_balancing_table,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  warnOnNonHashBasedMetrics(metrics);

  auto [from_table, to_table] = columnMapsFromAssociateSamples(
      rlhf_samples, _featurizer->textDatasetConfig());

  from_table =
      _featurizer->inferInputTransform()->apply(std::move(from_table), *_state);
  to_table =
      _featurizer->inferInputTransform()->apply(std::move(to_table), *_state);

  return py::cast(_mach->associate(
      std::move(from_table), to_table, std::move(featurized_balancing_table),
      inputColumns(), docIdColumn(), learning_rate,
      /* repeats= */ n_association_samples,
      /* num_buckets= */ n_buckets, /* epochs= */ epochs,
      /* batch_size= */ options.batchSize(),
      /* metrics= */ getMetrics(metrics, "train_"),
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval));
}

void UDTMachClassifier::associate(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
  auto balancing_table = rlhfBalancerTable(
      /* sampler= */ *_state->rlhfSampler(),
      /* text_column_name= */ _featurizer->textDatasetConfig().textColumn(),
      /* id_column_name= */ docIdColumn(),
      /* input_transform= */ *_featurizer->inferInputTransform(),
      /* state= */ *_state,
      /* num_samples= */ rlhf_samples.size() * n_balancing_samples);

  TrainOptions options;
  options.batch_size = defaults::ASSOCIATE_BATCH_SIZE;
  associateTrain(balancing_table, rlhf_samples, n_buckets,
                 n_association_samples, learning_rate, epochs,
                 /* metrics= */ {}, /* options= */ options);
}

py::object UDTMachClassifier::associateTrain(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto table = feat::CsvIterator::all(balancing_data, _delimiter);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);
  table = _featurizer->labelTransform()->apply(std::move(table), *_state);
  return associateTrain(std::move(table), rlhf_samples, n_buckets,
                        n_association_samples, learning_rate, epochs, metrics,
                        options);
}

py::object UDTMachClassifier::associateColdStart(
    const dataset::DataSourcePtr& balancing_data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  auto table = feat::CsvIterator::all(balancing_data, _delimiter);
  table = _featurizer
              ->unsupAugmenter(strong_column_names, weak_column_names,
                               /* fast_approximation= */ false)
              ->apply(std::move(table), *_state);
  table = _featurizer->inferInputTransform()->apply(std::move(table), *_state);
  table = _featurizer->labelTransform()->apply(std::move(table), *_state);
  return associateTrain(std::move(table), rlhf_samples, n_buckets,
                        n_association_samples, learning_rate, epochs, metrics,
                        options);
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

void UDTMachClassifier::setIndex(const dataset::mach::MachIndexPtr& index) {
  _mach->setIndex(index);
}

void UDTMachClassifier::setMachSamplingThreshold(float threshold) {
  _mach->setMachSamplingThreshold(threshold);
}

InputMetrics UDTMachClassifier::getMetrics(
    const std::vector<std::string>& metric_names, const std::string& prefix) {
  const auto& model = _mach->model();
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
  archive(cereal::base_class<UDTBackend>(this), _mach, _delimiter, _featurizer,
          _state, _default_top_k_to_return, _num_buckets_to_eval);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)