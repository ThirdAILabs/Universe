#include "UDTMultiMach.h"
#include <bolt/src/train/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/InputTypes.h>
#include <pybind11/stl.h>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::udt {

UDTMultiMach::UDTMultiMach(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const CategoricalDataTypePtr& target,
    uint32_t n_target_classes, bool integer_target,
    const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    config::ArgumentMap user_args) {
  size_t n_models = user_args.get<uint32_t>("n_models", "integer");

  uint32_t num_buckets =
      user_args.get<uint32_t>("extreme_output_dim", "integer",
                              UDTMach::autotuneMachOutputDim(n_target_classes));
  user_args.insert<uint32_t>("extreme_output_dim", num_buckets / n_models);
  user_args.insert<uint32_t>("extreme_num_hashes", 1);
  user_args.insert("softmax", true);

  for (size_t i = 0; i < n_models; i++) {
    _models.push_back(std::make_unique<UDTMach>(
        input_data_types, temporal_tracking_relationships, target_name, target,
        n_target_classes, integer_target, tabular_options, model_config,
        user_args));
    _models.back()->getIndex()->setSeed((i + 17) * 113);
  }
}

class MultiMachTrainLogger final : public bolt::callbacks::Callback {
 public:
  explicit MultiMachTrainLogger(uint32_t model_id) : _model_id(model_id) {}

  void onEpochEnd() final {
    uint32_t epoch = model->epochs();

    std::cout << "Model " << _model_id << " | train | epoch " << epoch
              << " | total_train_steps " << model->trainSteps();

    if (history->size() > 1) {
      std::cout << " |";
    }
    for (const auto& [name, values] : *history) {
      if (name != "epoch_times") {
        std::cout << " " << name << "=" << values.back();
      }
    }
    std::cout << " | time " << history->at("epoch_times").back() << std::endl;
  }

 private:
  uint32_t _model_id;
};

std::shared_ptr<MultiMachTrainLogger> logger(uint32_t model_id) {
  return std::make_shared<MultiMachTrainLogger>(model_id);
}

py::object UDTMultiMach::train(const dataset::DataSourcePtr& data,
                               float learning_rate, uint32_t epochs,
                               const std::vector<std::string>& train_metrics,
                               const dataset::DataSourcePtr& val_data,
                               const std::vector<std::string>& val_metrics,
                               const std::vector<CallbackPtr>& callbacks,
                               TrainOptions options,
                               const bolt::DistributedCommPtr& comm) {
  if (comm) {
    throw std::invalid_argument("Cannot pass 'comm' to MultiMach.");
  }
  if (val_data || !val_metrics.empty()) {
    throw std::invalid_argument("Validation is not supported for MultiMach.");
  }

  options.verbose = false;

  py::object metrics;
  uint32_t model_id = 0;
  for (auto& model : _models) {
    std::vector<CallbackPtr> model_callbacks;
    if (model_id == 0) {
      model_callbacks = callbacks;
    }
    model_callbacks.push_back(logger(model_id++));

    metrics = model->train(data, learning_rate, epochs, train_metrics, nullptr,
                           {}, model_callbacks, options, nullptr);
    data->restart();
    std::cout << std::endl;
  }

  return metrics;
}

py::object UDTMultiMach::coldstart(
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
  if (comm) {
    throw std::invalid_argument("Cannot pass 'comm' to MultiMach.");
  }
  if (val_data || !val_metrics.empty()) {
    throw std::invalid_argument("Validation is not supported for MultiMach.");
  }

  options.verbose = false;

  py::object metrics;
  uint32_t model_id = 0;
  for (auto& model : _models) {
    std::vector<CallbackPtr> model_callbacks;
    if (model_id == 0) {
      model_callbacks = callbacks;
    }
    model_callbacks.push_back(logger(model_id++));

    metrics = model->coldstart(
        /*data=*/data, /*strong_column_names=*/strong_column_names,
        /*weak_column_names=*/weak_column_names,
        /*variable_length=*/variable_length, /*learning_rate=*/learning_rate,
        /*epochs=*/epochs, /*train_metrics=*/train_metrics,
        /*val_data=*/nullptr, /*val_metrics=*/{}, /*callbacks=*/model_callbacks,
        /*options=*/options, /*comm=*/nullptr);
    data->restart();
    std::cout << std::endl;
  }

  return metrics;
}

class MultiMachMetric {
 public:
  static MultiMachMetric precision(size_t k) {
    return MultiMachMetric(k, Type::Precision);
  }

  static MultiMachMetric recall(size_t k) {
    return MultiMachMetric(k, Type::Recall);
  }

  void record(const std::vector<Scores>& scores,
              const data::ArrayColumnBasePtr<uint32_t>& label_batch) {
    for (size_t i = 0; i < scores.size(); i++) {
      auto n_candidates = std::min(_k, scores[i].size());
      auto labels = label_batch->row(i);
      for (size_t j = 0; j < n_candidates; j++) {
        if (std::find(labels.begin(), labels.end(), scores[i][j].first) !=
            labels.end()) {
          _true_positives++;
        }
      }

      switch (_type) {
        case Type::Precision:
          _total += n_candidates;
          break;
        case Type::Recall:
          _total += labels.size();
          break;
        default:
          throw std::runtime_error("Unhandled metric type.");
      }
    }
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
  enum class Type { Precision, Recall };

  MultiMachMetric(size_t k, Type type) : _k(k), _type(type) {}

  size_t _true_positives = 0;
  size_t _total = 0;

  size_t _k;

  Type _type;
};

uint32_t parseK(const std::string& metric_name) {
  std::smatch k_match;
  std::regex_search(metric_name, k_match, std::regex("[1-9]\\d*"));
  return std::stoul(metric_name.substr(k_match.position(), k_match.length()));
}

std::pair<std::vector<MultiMachMetric>, uint32_t> getMetricTrackers(
    const std::vector<std::string>& metrics) {
  uint32_t top_k_for_metrics = 0;
  std::vector<MultiMachMetric> metric_trackers;
  for (const auto& metric : metrics) {
    if (std::regex_match(metric, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = parseK(metric);
      metric_trackers.push_back(MultiMachMetric::precision(k));
      top_k_for_metrics = std::max(top_k_for_metrics, k);
    } else if (std::regex_match(metric, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = parseK(metric);
      metric_trackers.push_back(MultiMachMetric::recall(k));
      top_k_for_metrics = std::max(top_k_for_metrics, k);
    }
  }

  return {metric_trackers, top_k_for_metrics};
}

py::object UDTMultiMach::evaluate(const dataset::DataSourcePtr& data,
                                  const std::vector<std::string>& metrics,
                                  bool sparse_inference, bool verbose,
                                  std::optional<uint32_t> top_k) {
  (void)top_k;
  (void)verbose;

  auto [metric_trackers, top_k_for_metrics] = getMetricTrackers(metrics);

  for (auto& model : _models) {
    model->setDecodeParams(_num_buckets_to_eval, _num_buckets_to_eval);
  }

  auto text_col = _models.at(0)->textDatasetConfig().textColumn();
  auto label_col = _models.at(0)->textDatasetConfig().labelColumn();
  auto label_delim = _models.at(0)->textDatasetConfig().labelDelimiter();

  data::TransformationPtr parse_labels;
  if (label_delim) {
    parse_labels = std::make_shared<data::StringToTokenArray>(
        label_col, label_col, label_delim.value(), std::nullopt);
  } else {
    parse_labels = std::make_shared<data::StringToToken>(label_col, label_col,
                                                         std::nullopt);
  }

  auto data_iter =
      data::CsvIterator::make(data, _models.at(0)->featurizer()->delimiter());

  while (auto batch = data_iter->next()) {
    size_t batch_size = batch->numRows();

    auto text_data = batch->getValueColumn<std::string>(text_col);
    auto parsed_batch = parse_labels->applyStateless(*batch);
    auto label_data = parsed_batch.getArrayColumn<uint32_t>(label_col);

    MapInputBatch input_batch(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      input_batch[i] = {{text_col, text_data->value(i)}};
    }

    auto scores = predictImpl(input_batch, sparse_inference, top_k_for_metrics);

    for (auto& tracker : metric_trackers) {
      tracker.record(scores, label_data);
    }
  }

  std::cout << "validate | epoch " << _models[0]->model()->epochs()
            << " | total_train_steps " << _models[0]->model()->trainSteps()
            << " | ";

  bolt::metrics::History output_metrics;
  for (const auto& tracker : metric_trackers) {
    std::cout << tracker.name() << "=" << tracker.value() << " ";
    output_metrics["val_" + tracker.name()].push_back(tracker.value());
  }
  std::cout << std::endl;

  return py::cast(output_metrics);
}

py::object UDTMultiMach::predict(const MapInput& sample, bool sparse_inference,
                                 bool return_predicted_class,
                                 std::optional<uint32_t> top_k) {
  (void)return_predicted_class;

  return py::cast(predictImpl({sample}, sparse_inference,
                              top_k.value_or(_default_top_k_to_return))[0]);
}

py::object UDTMultiMach::predictBatch(const MapInputBatch& samples,
                                      bool sparse_inference,
                                      bool return_predicted_class,
                                      std::optional<uint32_t> top_k) {
  (void)return_predicted_class;

  return py::cast(predictImpl(samples, sparse_inference,
                              top_k.value_or(_default_top_k_to_return)));
}

std::vector<Scores> UDTMultiMach::predictImpl(const MapInputBatch& input,
                                              bool sparse_inference,
                                              uint32_t top_k) {
  if (_fast_decode) {
    return predictFastDecode(input, sparse_inference, top_k);
  }
  return predictRegularDecode(input, sparse_inference, top_k);
}

std::vector<Scores> UDTMultiMach::predictFastDecode(const MapInputBatch& input,
                                                    bool sparse_inference,
                                                    uint32_t top_k) {
  std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>> scores;
  for (auto& model : _models) {
    auto preds =
        model->predictBatchImpl(input, sparse_inference, false, std::nullopt);
    scores.push_back(preds);
  }

  std::vector<Scores> output(input.size());

#pragma omp parallel for default(none) \
    shared(input, scores, output, top_k) if (input.size() > 1)
  for (size_t i = 0; i < input.size(); i++) {
    std::unordered_map<uint32_t, float> sample_scores;
    for (const auto& model_scores : scores) {
      for (const auto& [id, score] : model_scores.at(i)) {
        sample_scores[id] += score;
      }
    }
    Scores results(sample_scores.begin(), sample_scores.end());
    std::sort(results.begin(), results.end(), BestScore{});
    if (results.size() > top_k) {
      results.resize(top_k);
    }
    output[i] = results;
  }

  return output;
}

std::vector<Scores> UDTMultiMach::predictRegularDecode(
    const MapInputBatch& input, bool sparse_inference, uint32_t top_k) {
  bolt::TensorList scores;
  for (auto& model : _models) {
    auto output = model->model()->forward(
        model->featurizer()->featurizeInputBatch(input), sparse_inference);
    scores.push_back(output.at(0));
  }

  std::vector<Scores> output(input.size());

  for (size_t i = 0; i < input.size(); i++) {
    std::unordered_map<uint32_t, float> candidate_scores;
    for (size_t m = 0; m < _models.size(); m++) {
      const auto& index = _models[m]->getIndex();
      auto top_model_candidates =
          scores[m]->getVector(i).findKLargestActivations(_num_buckets_to_eval);

      while (!top_model_candidates.empty()) {
        uint32_t bucket = top_model_candidates.top().second;
        for (uint32_t id : index->getEntities(bucket)) {
          candidate_scores[id] = 0.0;
        }
        top_model_candidates.pop();
      }
    }

    for (size_t m = 0; m < _models.size(); m++) {
      const auto& index = _models[m]->getIndex();
      const BoltVector& score_vec = scores[m]->getVector(i);
      if (score_vec.isDense()) {
        const float* activations = scores[m]->getVector(i).activations;
        for (auto& [id, score] : candidate_scores) {
          score += activations[index->getHashes(id)[0]];
        }
      } else {
        std::unordered_map<uint32_t, float> score_map;
        for (size_t j = 0; j < score_vec.len; j++) {
          score_map[score_vec.active_neurons[j]] = score_vec.activations[j];
        }
        for (auto& [id, score] : candidate_scores) {
          uint32_t hash = index->getHashes(id)[0];
          if (score_map.count(hash)) {
            score += score_map[hash];
          }
        }
      }
    }

    Scores results(candidate_scores.begin(), candidate_scores.end());
    std::sort(results.begin(), results.end(), BestScore{});
    if (results.size() > top_k) {
      results.resize(top_k);
    }
    output[i] = results;
  }

  return output;
}

ar::ConstArchivePtr UDTMultiMach::toArchive(bool with_optimizer) const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("default_top_k_to_return", ar::u64(_default_top_k_to_return));
  map->set("num_buckets_to_eval", ar::u64(_num_buckets_to_eval));

  auto models = ar::List::make();
  for (const auto& model : _models) {
    models->append(model->toArchive(with_optimizer));
  }

  map->set("models", models);

  return map;
}

UDTMultiMach::UDTMultiMach(const ar::Archive& archive)
    : _default_top_k_to_return(archive.u64("default_top_k_to_return")),
      _num_buckets_to_eval(archive.u64("num_buckets_to_eval")) {
  for (const auto& model_archive : archive.get("models")->list()) {
    _models.push_back(UDTMach::fromArchive(*model_archive));
  }
}

std::unique_ptr<UDTMultiMach> UDTMultiMach::fromArchive(
    const ar::Archive& archive) {
  return std::make_unique<UDTMultiMach>(archive);
}

}  // namespace thirdai::automl::udt