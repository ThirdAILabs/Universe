#pragma once

#include <bolt/src/nn/model/Model.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Transformation.h>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::NER {
using PerTokenPredictions = std::vector<std::pair<uint32_t, float>>;
using PerTokenListPredictions = std::vector<PerTokenPredictions>;

static std::vector<PerTokenListPredictions> getTags(
    std::vector<std::vector<std::string>> tokens, uint32_t top_k,
    std::string tokens_column, const data::TransformationPtr& inference_transform,
    const data::OutputColumnsList& bolt_inputs,
    const bolt::ModelPtr& bolt_model) {
  std::vector<PerTokenListPredictions> tags_and_scores;
  tags_and_scores.reserve(tokens.size());

  for (const auto& sub_vector : tokens) {
    PerTokenListPredictions predictions(sub_vector.size());
    tags_and_scores.push_back(predictions);
  }
  data::ColumnMap data(
      data::ColumnMap({{tokens_column, data::ArrayColumn<std::string>::make(
                                           std::move(tokens), std::nullopt)}}));

  // featurize input data
  auto columns = inference_transform->applyStateless(data);
  auto tensors = data::toTensorBatches(columns, bolt_inputs, 2048);

  size_t sub_vector_index = 0;
  size_t token_index = 0;

  for (const auto& batch : tensors) {
    auto outputs = bolt_model->forward(batch).at(0);

    for (size_t i = 0; i < outputs->batchSize(); i += 1) {
      if (token_index >= tags_and_scores[sub_vector_index].size()) {
        token_index = 0;
        sub_vector_index++;
      }
      auto token_level_predictions = outputs->getVector(i).topKNeurons(top_k);
      while (!token_level_predictions.empty()) {
        float score = token_level_predictions.top().first;
        uint32_t tag = token_level_predictions.top().second;
        tags_and_scores[sub_vector_index][token_index].push_back({tag, score});
        token_level_predictions.pop();
      }
      // topkactivation is a min heap hence, reverse it
      std::reverse(tags_and_scores[sub_vector_index][token_index].begin(),
                   tags_and_scores[sub_vector_index][token_index].end());
      if (sub_vector_index >= tags_and_scores.size()) {
        throw std::runtime_error("tags indices not matching");
      }
      token_index += 1;
    }
  }
  return tags_and_scores;
}

inline uint32_t getMaxLabelFromTagToLabel(
    std::unordered_map<std::string, uint32_t>& tag_to_label) {
  auto maxPair = std::max_element(
      tag_to_label.begin(), tag_to_label.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  return maxPair->second + 1;
}

// Clang-tidy wants this function inline, which shouldnt be inlined
inline std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
getNerTagsFromTokens(
    std::unordered_map<uint32_t, std::string> label_to_tag_map,
    const std::vector<PerTokenListPredictions>& tags_and_scores) {
  std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
      string_and_scores;

  for (const auto& sentence_tags_and_scores : tags_and_scores) {
    std::vector<std::vector<std::pair<std::string, float>>>
        sentence_string_tags_and_scores;
    sentence_string_tags_and_scores.reserve(sentence_tags_and_scores.size());
    for (const auto& tags_and_scores : sentence_tags_and_scores) {
      std::vector<std::pair<std::string, float>> token_tags_and_scores;
      token_tags_and_scores.reserve(tags_and_scores.size());
      for (const auto& tag_and_score : tags_and_scores) {
        token_tags_and_scores.push_back(
            {label_to_tag_map[tag_and_score.first], tag_and_score.second});
      }
      sentence_string_tags_and_scores.push_back(token_tags_and_scores);
    }
    string_and_scores.push_back(sentence_string_tags_and_scores);
  }
  return string_and_scores;
}

}  // namespace thirdai::bolt::NER
