#include "UDTMultiMach.h"
#include <bolt/src/train/metrics/Metric.h>
#include <_types/_uint32_t.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/InputTypes.h>
#include <stdexcept>
#include <string>
#include <pybind11/stl.h>
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
    _models.emplace_back(input_data_types, temporal_tracking_relationships,
                         target_name, target, n_target_classes, integer_target,
                         tabular_options, model_config, user_args);
    _models.back().getIndex()->setSeed((i + 17) * 113);
  }
}

py::object UDTMultiMach::train(const dataset::DataSourcePtr& data,
                               float learning_rate, uint32_t epochs,
                               const std::vector<std::string>& train_metrics,
                               const dataset::DataSourcePtr& val_data,
                               const std::vector<std::string>& val_metrics,
                               const std::vector<CallbackPtr>& callbacks,
                               TrainOptions options,
                               const bolt::DistributedCommPtr& comm) {
  if (!callbacks.empty()) {
    throw std::invalid_argument("Cannot pass 'callbacks' to MultiMach.");
  }
  if (comm) {
    throw std::invalid_argument("Cannot pass 'comm' to MultiMach.");
  }

  py::object metrics;
  for (auto& model : _models) {
    metrics = model.train(data, learning_rate, epochs, train_metrics, val_data,
                          val_metrics, {}, options, nullptr);
    data->restart();
    if (val_data) {
      val_data->restart();
    }
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
  if (!callbacks.empty()) {
    throw std::invalid_argument("Cannot pass 'callbacks' to MultiMach.");
  }
  if (comm) {
    throw std::invalid_argument("Cannot pass 'comm' to MultiMach.");
  }

  py::object metrics;
  for (auto& model : _models) {
    metrics =
        model.coldstart(data, strong_column_names, weak_column_names,
                        variable_length, learning_rate, epochs, train_metrics,
                        val_data, val_metrics, {}, options, nullptr);
    data->restart();
    if (val_data) {
      val_data->restart();
    }
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
    model.setDecodeParams(_num_buckets_to_eval, _num_buckets_to_eval);
  }

  auto text_col = _models.at(0).textDatasetConfig().textColumn();
  auto label_col = _models.at(0).textDatasetConfig().labelColumn();
  auto label_delim = _models.at(0).textDatasetConfig().labelDelimiter();

  data::TransformationPtr parse_labels;
  if (label_delim) {
    parse_labels = std::make_shared<data::StringToTokenArray>(
        label_col, label_col, label_delim.value(), std::nullopt);
  } else {
    parse_labels = std::make_shared<data::StringToToken>(label_col, label_col,
                                                         std::nullopt);
  }

  auto data_iter =
      data::CsvIterator::make(data, _models.at(0).featurizer()->delimiter());

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

    std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>> scores;
    for (auto& model : _models) {
      auto preds = model.predictBatchImpl(input_batch, sparse_inference, false,
                                          std::nullopt);
      scores.push_back(preds);
    }

    for (size_t i = 0; i < batch_size; i++) {
      uint32_t best_id;
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

      auto labels = label_data->row(i);
      if (std::find(labels.begin(), labels.end(), best_id) != labels.end()) {
        correct += 1;
      }
      total += 1;
    }
  }

  std::cerr << "precision@1 = " << (static_cast<double>(correct) / total)
            << std::endl;
  std::unordered_map<std::string, double> output = {
      {"precision@1", static_cast<double>(correct) / total}};
  return py::cast(output);
}

}  // namespace thirdai::automl::udt