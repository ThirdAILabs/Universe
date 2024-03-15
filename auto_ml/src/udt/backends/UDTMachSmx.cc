#include "UDTMachSmx.h"
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/utils/Timer.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/UDTMach.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/SmxTensorConversion.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/TextTokenizer.h>
#include <pybind11/stl.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/metrics/Metrics.h>
#include <smx/src/optimizers/Adam.h>
#include <smx/src/tensor/DenseTensor.h>
#include <utils/Logging.h>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::udt {

UDTMachSmx::UDTMachSmx(
    ColumnDataTypes input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const CategoricalDataTypePtr& target,
    uint32_t n_target_classes, bool integer_target,
    const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& args)
    : _delimiter(tabular_options.delimiter),
      _softmax(args.get<bool>("softmax", "bool", false)) {
  input_data_types.erase(target_name);
  CHECK(input_data_types.size() == 1, "Expected only input data type.");
  auto text_name = input_data_types.begin()->first;
  auto text = asText(input_data_types.begin()->second);
  CHECK(text, "Expected input data type to be text.");

  CHECK(temporal_tracking_relationships.empty(),
        "Temporal tracking not supported.");
  CHECK(integer_target, "Integer target must be true.");
  CHECK(!model_config, "Model config must be false.");

  size_t input_dim = tabular_options.feature_hash_range;

  size_t num_buckets =
      args.get<uint32_t>("extreme_output_dim", "integer",
                         UDTMach::autotuneMachOutputDim(n_target_classes));

  bool hidden_bias =
      args.get<bool>("hidden_bias", "bool", defaults::HIDDEN_BIAS);
  bool output_bias =
      args.get<bool>("output_bias", "bool", defaults::OUTPUT_BIAS);

  _model = std::make_unique<MachModel>(
      input_dim,
      args.get<uint32_t>("embedding_dimension", "integer",
                         defaults::HIDDEN_DIM),
      num_buckets, utils::autotuneSparsity(num_buckets), hidden_bias,
      output_bias);

  _optimizer = std::make_unique<smx::Adam>(_model->parameters(), 1e-3);
  _optimizer->registerOnUpdateCallback(_model->out->onUpdateCallback());

  _freeze_hash_tables = args.get<bool>("freeze_hash_tables", "boolean",
                                       defaults::FREEZE_HASH_TABLES);

  size_t num_hashes = args.get<uint32_t>(
      "extreme_num_hashes", "integer",
      UDTMach::autotuneMachNumHashes(n_target_classes, num_buckets));

  _mach_index = dataset::mach::MachIndex::make(
      /* num_buckets = */ num_buckets, /* num_hashes = */ num_hashes,
      /* num_elements = */ n_target_classes);

  _text_transform = std::make_shared<data::TextTokenizer>(
      text_name, FEATURIZED_INDICES, FEATURIZED_VALUES, text->tokenizer,
      text->encoder, text->lowercase, input_dim);

  if (target->delimiter) {
    _entity_parse_transform = std::make_shared<data::StringToTokenArray>(
        target_name, MACH_DOC_IDS, *target->delimiter, std::nullopt);
  } else {
    _entity_parse_transform = std::make_shared<data::StringToToken>(
        target_name, MACH_DOC_IDS, std::nullopt);
  }

  _mach_label_transform =
      std::make_shared<data::MachLabel>(MACH_DOC_IDS, MACH_LABELS);

  _input_columns = {data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)};
  if (_softmax) {
    _label_columns = {
        data::OutputColumns(MACH_LABELS, data::ValueFillType::SumToOne)};
  } else {
    _label_columns = {
        data::OutputColumns(MACH_LABELS, data::ValueFillType::Ones)};
  }
}

py::object UDTMachSmx::train(const dataset::DataSourcePtr& data,
                             float learning_rate, uint32_t epochs,
                             const std::vector<std::string>& train_metrics,
                             const dataset::DataSourcePtr& val_data,
                             const std::vector<std::string>& val_metrics,
                             const std::vector<CallbackPtr>& callbacks,
                             TrainOptions options,
                             const bolt::DistributedCommPtr& comm) {
  CHECK(train_metrics.empty(), "Arg 'train_metrics' not supported.");
  CHECK(!val_data, "Arg 'val_data' not supported");
  CHECK(val_metrics.empty(), "Arg 'val_metrics' not supported");
  CHECK(callbacks.empty(), "Arg 'callbacks' not supported");
  CHECK(!options.max_in_memory_batches, "Arg 'max_in_memory' not supported");
  CHECK(!comm, "Arg 'comm' not supported");

  auto dataset =
      loadTrainingData(data, options.batchSize(), options.shuffle_config);

  bolt::metrics::History metrics;
  for (uint32_t end = _epoch + epochs; _epoch < end; _epoch++) {
    if (_freeze_hash_tables && _epoch == 1) {
      _model->out->neuronIndex()->freeze(true);
    }
    bolt::utils::Timer timer;
    train(dataset, learning_rate);
    timer.stop();

    metrics["epoch_times"].push_back(timer.seconds());
    if (options.verbose) {
      std::cout << "train | epoch " << _epoch << " | steps " << dataset.size()
                << " | time " << timer.seconds() << "s\n"
                << std::endl;
    }
  }

  return py::cast(metrics);
}

py::object UDTMachSmx::coldstart(
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
  CHECK(train_metrics.empty(), "Arg 'train_metrics' not supported.");
  CHECK(!val_data, "Arg 'val_data' not supported");
  CHECK(val_metrics.empty(), "Arg 'val_metrics' not supported");
  CHECK(callbacks.empty(), "Arg 'callbacks' not supported");
  CHECK(!options.max_in_memory_batches, "Arg 'max_in_memory' not supported");
  CHECK(!comm, "Arg 'comm' not supported");

  auto cold_start =
      coldStartTransform(strong_column_names, weak_column_names,
                         variable_length, /*fast_approximation=*/false);

  bolt::metrics::History metrics;
  for (uint32_t end = _epoch + epochs; _epoch < end; _epoch++) {
    if (_freeze_hash_tables && _epoch == 1) {
      _model->out->neuronIndex()->freeze(true);
    }
    auto dataset = loadTrainingData(data, options.batchSize(),
                                    options.shuffle_config, cold_start);

    bolt::utils::Timer timer;
    train(dataset, learning_rate);
    timer.stop();
    metrics["epoch_times"].push_back(timer.seconds());

    if (options.verbose) {
      std::cout << "coldstart | epoch " << _epoch << " | steps "
                << dataset.size() << " | time " << timer.seconds() << "s\n"
                << std::endl;
    }
  }

  return py::cast(metrics);
}

class MachMetric {
 public:
  MachMetric(const std::string& name, data::MachIndexPtr index,
             uint32_t default_num_buckets_to_eval)
      : _index(std::move(index)),
        _default_num_buckets_to_eval(default_num_buckets_to_eval) {
    if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      _k = parseK(name);
      _type = Type::Precision;
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      _k = parseK(name);
      _type = Type::Recall;
    } else {
      throw std::invalid_argument("Unsupported metric '" + name + "'.");
    }
  }

  void record(const smx::DenseTensorPtr& scores,
              const std::vector<std::vector<uint32_t>>& labels) {
    size_t n_buckets = _index->numBuckets();
    float* data = scores->data<float>();

    size_t true_positives = 0, total = 0;

#pragma omp parallel for default(none) shared(labels, data, n_buckets), reduction(+: true_positives, total)
    for (size_t i = 0; i < labels.size(); i++) {
      BoltVector vec(/*an=*/nullptr, /*a=*/data + i * n_buckets, /*g=*/nullptr,
                     /*l=*/n_buckets);
      auto topk_scores = _index->decode(vec, _k, _default_num_buckets_to_eval);

      for (const auto& [id, score] : topk_scores) {
        if (std::find(labels[i].begin(), labels[i].end(), id) !=
            labels[i].end()) {
          true_positives++;
        }
      }

      switch (_type) {
        case Type::Precision:
          total += topk_scores.size();
          break;
        case Type::Recall:
          total += labels[i].size();
          break;
      }
    }

    _true_positives += true_positives;
    _total += total;
  }

  float value() const { return static_cast<float>(_true_positives) / _total; }

  std::string name() const {
    switch (_type) {
      case Type::Precision:
        return "precision@" + std::to_string(_k);
      case Type::Recall:
        return "recall@" + std::to_string(_k);
      default:
        throw std::runtime_error("Unhandled metric type.");
    }
  }

 private:
  static uint32_t parseK(const std::string& metric_name) {
    std::smatch k_match;
    std::regex_search(metric_name, k_match, std::regex("[1-9]\\d*"));
    return std::stoul(metric_name.substr(k_match.position(), k_match.length()));
  }

  enum class Type { Precision, Recall };

  data::MachIndexPtr _index;
  uint32_t _k;
  uint32_t _default_num_buckets_to_eval;
  Type _type;

  size_t _true_positives = 0;
  size_t _total = 0;
};

py::object UDTMachSmx::evaluate(const dataset::DataSourcePtr& data,
                                const std::vector<std::string>& metrics,
                                bool sparse_inference, bool verbose,
                                std::optional<uint32_t> top_k) {
  CHECK(!sparse_inference, "Sparse inference is not yet supported.");
  (void)top_k;

  std::vector<MachMetric> metric_trackers;
  metric_trackers.reserve(metrics.size());
  for (const auto& metric : metrics) {
    metric_trackers.push_back(
        MachMetric(metric, _mach_index, _default_num_buckets_to_eval));
  }

  auto dataset = loadEvalData(data, defaults::BATCH_SIZE);

  _model->eval();

  bolt::utils::Timer timer;
  for (const auto& [input, labels] : dataset) {
    auto output = _model->forward(input);
    auto scores = getScores(output->tensor());

    for (auto& metric : metric_trackers) {
      metric.record(scores, labels);
    }
  }
  timer.stop();

  bolt::metrics::History output;
  for (const auto& metric : metric_trackers) {
    output["val_" + metric.name()].push_back(metric.value());
  }
  output["val_times"].push_back(timer.seconds());

  if (verbose) {
    std::cout << "eval | eval_batches " << dataset.size() << " | ";
  }
  for (const auto& metric : metric_trackers) {
    std::cout << metric.name() << "=" << metric.value() << " ";
  }
  std::cout << "| time " << timer.seconds() << "s\n" << std::endl;

  return py::cast(output);
}

py::object UDTMachSmx::predict(const MapInput& sample, bool sparse_inference,
                               bool return_predicted_class,
                               std::optional<uint32_t> top_k) {
  CHECK(!sparse_inference, "Sparse inference is not yet supported.");
  CHECK(!return_predicted_class, "return_predicted_class is not supported.");

  _model->eval();

  auto input = featurizeInput(data::ColumnMap::fromMapInput(sample));
  auto output = _model->forward(input);
  auto scores = getScores(output->tensor());

  float* data = scores->data<float>();

  return py::cast(decode(data, top_k.value_or(_default_topk)));
}

py::object UDTMachSmx::predictBatch(const MapInputBatch& sample,
                                    bool sparse_inference,
                                    bool return_predicted_class,
                                    std::optional<uint32_t> top_k) {
  CHECK(!sparse_inference, "Sparse inference is not yet supported.");
  CHECK(!return_predicted_class, "return_predicted_class is not supported.");

  _model->eval();

  auto input = featurizeInput(data::ColumnMap::fromMapInputBatch(sample));
  auto output = _model->forward(input);
  auto scores = getScores(output->tensor());

  float* data = scores->data<float>();
  size_t batch_size = scores->shape(0);
  size_t dim = scores->shape(1);

  uint32_t topk = top_k.value_or(_default_topk);

  std::vector<std::vector<std::pair<uint32_t, double>>> predictions(batch_size);

  for (size_t i = 0; i < batch_size; i++) {
    predictions[i] = decode(data + i * dim, topk);
  }

  return py::cast(predictions);
}

data::TransformationPtr UDTMachSmx::coldStartTransform(
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols,
    std::optional<data::VariableLengthConfig> variable_length,
    bool fast_approximation) const {
  if (fast_approximation) {
    std::vector<std::string> all_columns = weak_cols;
    all_columns.insert(all_columns.end(), strong_cols.begin(),
                       strong_cols.end());
    return std::make_shared<data::StringConcat>(all_columns,
                                                _text_transform->inputColumn());
  }

  if (variable_length) {
    return std::make_shared<data::VariableLengthColdStart>(
        /* strong_column_names= */ strong_cols,
        /* weak_column_names= */ weak_cols,
        /* output_column_name= */ _text_transform->inputColumn(),
        /* config= */ *variable_length);
  }

  return std::make_shared<data::ColdStartTextAugmentation>(
      /* strong_column_names= */ strong_cols,
      /* weak_column_names= */ weak_cols,
      /* output_column_name= */ _text_transform->inputColumn());
}

UDTMachSmx::TrainingDataset UDTMachSmx::loadTrainingData(
    const dataset::DataSourcePtr& data_source, size_t batch_size,
    dataset::DatasetShuffleConfig shuffle_config,
    const data::TransformationPtr& cold_start) const {
  auto data_iter = data::CsvIterator::make(data_source, _delimiter);

  auto pipeline = data::Pipeline::make();
  if (cold_start) {
    pipeline = pipeline->then(cold_start);
  }
  pipeline = pipeline->then(_text_transform)
                 ->then(_entity_parse_transform)
                 ->then(_mach_label_transform);

  data::Loader loader(
      data_iter, pipeline, std::make_shared<data::State>(_mach_index),
      _input_columns, _label_columns, batch_size, /*shuffle=*/true,
      /*verbose=*/true, shuffle_config.min_buffer_size, shuffle_config.seed);

  auto [inputs, labels] = loader.allSmx();

  TrainingDataset dataset;
  for (size_t i = 0; i < inputs.size(); i++) {
    dataset.emplace_back(
        smx::Variable::make(inputs.at(i).at(0), /*requires_grad=*/false),
        smx::Variable::make(labels.at(i).at(0), /*requires_grad=*/false));
  }

  return dataset;
}

UDTMachSmx::EvalDataset UDTMachSmx::loadEvalData(
    const dataset::DataSourcePtr& data_source, size_t batch_size,
    const data::TransformationPtr& cold_start) const {
  auto columns = data::CsvIterator::all(data_source, _delimiter);

  auto pipeline = data::Pipeline::make();
  if (cold_start) {
    pipeline = pipeline->then(cold_start);
  }
  pipeline = pipeline->then(_text_transform)->then(_entity_parse_transform);

  columns = pipeline->applyStateless(columns);

  auto inputs = data::toSmxTensorBatches(columns, _input_columns, batch_size);

  auto labels = columns.getArrayColumn<uint32_t>(MACH_DOC_IDS);

  size_t row_cnt = 0;
  EvalDataset dataset;
  dataset.reserve(inputs.size());
  for (const auto& input : inputs) {
    size_t batch_size = input[0]->shape(0);

    std::vector<std::vector<uint32_t>> batch_labels;
    batch_labels.reserve(batch_size);
    for (size_t i = row_cnt; i < row_cnt + batch_size; i++) {
      batch_labels.push_back(labels->row(i).toVector());
    }
    row_cnt += batch_size;

    dataset.emplace_back(smx::Variable::make(input[0], /*requires_grad=*/false),
                         batch_labels);
  }

  return dataset;
}

void UDTMachSmx::train(const UDTMachSmx::TrainingDataset& dataset,
                       float learning_rate) {
  _model->train();

  _optimizer->updateLr(learning_rate);

  if (!dataset.empty()) {
    size_t batch_size = dataset.at(0).first->tensor()->shape(0);
    _model->out->autotuneHashTableRebuild(dataset.size(), batch_size);
  }

  size_t step = 0;
  for (const auto& [x, y] : dataset) {
    bolt::utils::Timer zero_timer;
    _optimizer->zeroGrad();

    zero_timer.stop();

    logging::info(fmt::format("smx zero_grad | epoch {} | step {} | time {} ms",
                              _epoch, step, zero_timer.milliseconds()));

    bolt::utils::Timer forward_backward_timer;
    auto out = _model->forward(x, y);

    if (_softmax) {
      auto loss = smx::crossEntropy(out, y);
      loss->backward();

    } else {
      auto loss = smx::binaryCrossEntropy(out, y);
      loss->backward();
    }

    forward_backward_timer.stop();
    logging::info(
        fmt::format("smx forward_backward | epoch {} | step {} | time {} ms",
                    _epoch, step, forward_backward_timer.milliseconds()));

    bolt::utils::Timer update_timer;
    _optimizer->step();
    update_timer.stop();
    logging::info(fmt::format("smx update | epoch {} | step {} | time {} ms",
                              _epoch, step, update_timer.milliseconds()));

    step++;
  }
}

}  // namespace thirdai::automl::udt