#include "UDTMultiMach.h"
#include <bolt/src/train/metrics/Metric.h>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/InputTypes.h>
#include <pybind11/stl.h>
#include <memory>
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

    metrics =
        model->coldstart(data, strong_column_names, weak_column_names,
                         variable_length, learning_rate, epochs, train_metrics,
                         nullptr, {}, model_callbacks, options, nullptr);
    data->restart();
    std::cout << std::endl;
  }

  return metrics;
}

py::object UDTMultiMach::evaluate(const dataset::DataSourcePtr& data,
                                  const std::vector<std::string>& metrics,
                                  bool sparse_inference, bool verbose,
                                  std::optional<uint32_t> top_k) {
  (void)top_k;
  (void)sparse_inference;
  (void)verbose;

  if (metrics.size() != 1 || metrics[0] != "precision@1") {
    throw std::invalid_argument("Metrics must be 'precision@1' for MultiMach.");
  }

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

  uint32_t correct = 0, total = 0;
  while (auto batch = data_iter->next()) {
    size_t batch_size = batch->numRows();

    auto text_data = batch->getValueColumn<std::string>(text_col);
    auto parsed_batch = parse_labels->applyStateless(*batch);
    auto label_data = parsed_batch.getArrayColumn<uint32_t>(label_col);

    MapInputBatch input_batch(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
      input_batch[i] = {{text_col, text_data->value(i)}};
    }

    std::vector<uint32_t> best_ids;
    if (_fast_decode) {
      best_ids = predictFastDecode(std::move(input_batch), sparse_inference);
    } else {
      best_ids = predictRegularDecode(std::move(input_batch));
    }

    for (size_t i = 0; i < batch_size; i++) {
      auto labels = label_data->row(i);
      if (std::find(labels.begin(), labels.end(), best_ids[i]) !=
          labels.end()) {
        correct += 1;
      }
      total += 1;
    }
  }

  float precision = static_cast<float>(correct) / total;

  std::cout << "validate | epoch " << _models[0]->model()->epochs()
            << " | total_train_steps " << _models[0]->model()->trainSteps()
            << " | val_precision@1=" << precision << "\n"
            << std::endl;
  bolt::metrics::History output = {{"val_precision@1", {precision}}};
  return py::cast(output);
}

std::vector<uint32_t> UDTMultiMach::predictFastDecode(MapInputBatch&& input,
                                                      bool sparse_inference) {
  std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>> scores;
  for (auto& model : _models) {
    auto preds =
        model->predictBatchImpl(input, sparse_inference, false, std::nullopt);
    scores.push_back(preds);
  }

  std::vector<uint32_t> preds(input.size());

#pragma omp parallel for default(none) shared(input, scores, preds)
  for (size_t i = 0; i < input.size(); i++) {
    uint32_t best_id = -1;
    double best_score = 0;
    std::unordered_map<uint32_t, double> sample_scores;
    for (const auto& model_scores : scores) {
      for (const auto& [id, score] : model_scores.at(i)) {
        sample_scores[id] += score;
        if (sample_scores[id] > best_score) {
          best_id = id;
          best_score = sample_scores[id];
        }
      }
    }
    preds[i] = best_id;
  }

  return preds;
}

std::vector<uint32_t> UDTMultiMach::predictRegularDecode(
    MapInputBatch&& input) {
  bolt::TensorList scores;
  for (auto& model : _models) {
    auto output = model->model()->forward(
        model->featurizer()->featurizeInputBatch(input), false);
    scores.push_back(output.at(0));
  }

  std::vector<uint32_t> preds(input.size());

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
    uint32_t best_id = -1;
    double best_score = 0;

    for (size_t m = 0; m < _models.size(); m++) {
      const auto& index = _models[m]->getIndex();
      const float* activations = scores[m]->getVector(i).activations;
      for (auto& [id, score] : candidate_scores) {
        score += activations[index->getHashes(id)[0]];
        if (score > best_score) {
          best_id = id;
          best_score = score;
        }
      }
    }
    preds[i] = best_id;
  }

  return preds;
}

ar::ConstArchivePtr UDTMultiMach::toArchive(bool with_optimizer) const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("top_k_to_return", ar::u64(_top_k_to_return));
  map->set("num_buckets_to_eval", ar::u64(_num_buckets_to_eval));

  auto models = ar::List::make();
  for (const auto& model : _models) {
    models->append(model->toArchive(with_optimizer));
  }

  map->set("models", models);

  return map;
}

UDTMultiMach::UDTMultiMach(const ar::Archive& archive)
    : _top_k_to_return(archive.u64("top_k_to_return")),
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