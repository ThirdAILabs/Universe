#include "UDTNer.h"
#include <auto_ml/src/udt/utils/KwargUtils.h>
#include <pybind11/stl.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace thirdai::automl::udt {

UDTNer::UDTNer(const ColumnDataTypes& data_types,
               const TokenTagsDataTypePtr& target,
               const std::string& target_name, const UDTNer* pretrained_model,
               const config::ArgumentMap& args)
    : _model(std::make_unique<NerModel>(
          data_types, target, target_name,
          pretrained_model ? pretrained_model->_model.get() : nullptr, args)) {}

py::object UDTNer::train(const dataset::DataSourcePtr& data,
                         float learning_rate, uint32_t epochs,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics,
                         const std::vector<CallbackPtr>& callbacks,
                         TrainOptions options,
                         const bolt::DistributedCommPtr& comm,
                         py::kwargs kwargs) {
  (void)kwargs;
  return py::cast(_model->train(data, learning_rate, epochs, train_metrics,
                                val_data, val_metrics, callbacks, options,
                                comm));
}

py::object UDTNer::evaluate(const dataset::DataSourcePtr& data,
                            const std::vector<std::string>& metrics,
                            bool sparse_inference, bool verbose,
                            py::kwargs kwargs) {
  (void)kwargs;
  return py::cast(_model->evaluate(data, metrics, sparse_inference, verbose));
}

std::vector<std::map<std::string, py::object>> formatResultsWithOffsets(
    const SentenceTags& tags,
    const std::vector<std::pair<size_t, size_t>>& offsets) {
  std::vector<std::map<std::string, py::object>> results;

  for (size_t i = 0; i < tags.size(); ++i) {
    if (!tags[i].empty()) {
      auto [start, end] = offsets[i];
      std::map<std::string, py::object> entity;
      entity["Score"] = py::cast(tags[i][0].second);
      entity["Type"] = py::cast(tags[i][0].first);
      entity["BeginOffset"] = py::cast(start);
      entity["EndOffset"] = py::cast(end);
      results.push_back(std::move(entity));
    }
  }

  return results;
}

py::object UDTNer::predict(const MapInput& sample, bool sparse_inference,
                           bool return_predicted_class,
                           std::optional<uint32_t> top_k,
                           const py::kwargs& kwargs) {
  (void)return_predicted_class;

  const auto& tokens_column = _model->getTokensColumn();

  if (!sample.count(tokens_column)) {
    throw std::invalid_argument("Expected input to contain column '" +
                                tokens_column + "'.");
  }

  float o_threshold =
      floatArg(kwargs, "threshold").value_or(defaults::NER_O_THRESHOLD);

  bool as_unicode = boolArg(kwargs, "as_unicode").value_or(false);

  auto [tags, offsets] =
      _model->predictTags({sample.at(tokens_column)}, sparse_inference,
                          top_k.value_or(1), o_threshold, as_unicode);

  if (kwargs.contains("return_offsets") &&
      py::cast<bool>(kwargs["return_offsets"])) {
    auto results = formatResultsWithOffsets(tags[0], offsets[0]);
    return py::cast(results);
  }

  return py::cast(tags[0]);
}

py::object UDTNer::predictBatch(const MapInputBatch& samples,
                                bool sparse_inference,
                                bool return_predicted_class,
                                std::optional<uint32_t> top_k,
                                const py::kwargs& kwargs) {
  (void)return_predicted_class;

  const auto& tokens_column = _model->getTokensColumn();

  std::vector<std::string> sentences;
  sentences.reserve(samples.size());
  for (const auto& sample : samples) {
    if (!sample.count(tokens_column)) {
      throw std::invalid_argument("Expected input to contain column '" +
                                  tokens_column + "'.");
    }

    sentences.push_back(sample.at(tokens_column));
  }

  float o_threshold =
      floatArg(kwargs, "threshold").value_or(defaults::NER_O_THRESHOLD);

  bool as_unicode = boolArg(kwargs, "as_unicode").value_or(false);

  auto [tags, offsets] = _model->predictTags(
      sentences, sparse_inference, top_k.value_or(1), o_threshold, as_unicode);

  if (kwargs.contains("return_offsets") &&
      py::cast<bool>(kwargs["return_offsets"])) {
    std::vector<std::vector<std::map<std::string, py::object>>> results_batch;
    for (size_t sentence_index = 0; sentence_index < tags.size();
         ++sentence_index) {
      auto results = formatResultsWithOffsets(tags[sentence_index],
                                              offsets[sentence_index]);
      results_batch.push_back(std::move(results));
    }
    return py::cast(results_batch);
  }

  return py::cast(tags);
}

ar::ConstArchivePtr UDTNer::toArchive(bool with_optimizer) const {
  return _model->toArchive(with_optimizer);
}

std::unique_ptr<UDTNer> UDTNer::fromArchive(const ar::Archive& archive) {
  return std::make_unique<UDTNer>(archive);
}

UDTNer::UDTNer(const ar::Archive& archive)
    : _model(NerModel::fromArchive(archive)) {}

}  // namespace thirdai::automl::udt
