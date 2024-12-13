#include "NerClassifier.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/ner/NerDyadicDataProcessor.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <utils/text/Stopwords.h>
#include <cctype>
#include <cstdint>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt::NER {

bool isAllPunctuation(const std::string& str) {
  return !str.empty() && std::all_of(str.begin(), str.end(), ::ispunct);
}

void applyPunctAndStopWordFilter(const std::string& token,
                                 PerTokenPredictions& predicted_tags,
                                 const std::string& default_tag) {
  // assumes that the highest activation vector is at the end

  auto cleaned_token = text::stripWhitespace(token);

  if (isAllPunctuation(cleaned_token) ||
      text::stop_words.count(cleaned_token)) {
    for (int i = predicted_tags.size() - 1; i >= 0; --i) {
      if (predicted_tags[i].first == default_tag) {
        predicted_tags[i].second = 1;
        std::rotate(predicted_tags.begin() + i, predicted_tags.begin() + i + 1,
                    predicted_tags.end());
        return;
      }
    }
    // If the tag 'O' is not found, then we erase the lowest activation tag and
    // insert 'O' as the highest activation tag.
    predicted_tags.push_back({default_tag, 1});
    predicted_tags.erase(predicted_tags.begin());
  }
}

metrics::History NerClassifier::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics) const {
  auto train_dataset =
      getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
  bolt::LabeledDataset val_dataset;
  if (val_data) {
    val_dataset =
        getDataLoader(val_data, batch_size, /* shuffle= */ false).all();
  }
  auto train_data_input = train_dataset.first;
  auto train_data_label = train_dataset.second;

  Trainer trainer(_bolt_model);
  trainer.train_with_metric_names(
      train_dataset, learning_rate, epochs, train_metrics, val_dataset,
      val_metrics, /* steps_per_validation= */ std::nullopt,
      /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ false, /* verbose= */ true);
  return trainer.getHistory();
}

data::Loader NerClassifier::getDataLoader(const dataset::DataSourcePtr& data,
                                          size_t batch_size,
                                          bool shuffle) const {
  auto data_iter =
      data::JsonIterator::make(data, {_tokens_column, _tags_column}, 1000);
  return data::Loader(data_iter, _train_transforms, nullptr, _bolt_inputs,
                      {data::OutputColumns(_tags_column)},
                      /* batch_size= */ batch_size,
                      /* shuffle= */ shuffle, /* verbose= */ true,
                      /* shuffle_buffer_size= */ 20000);
}

std::vector<PerTokenListPredictions> NerClassifier::getTags(
    std::vector<std::vector<std::string>> tokens, uint32_t top_k,
    const std::unordered_map<uint32_t, std::string>& label_to_tag_map,
    const std::unordered_map<std::string, uint32_t>& tag_to_label_map) const {
  std::vector<PerTokenListPredictions> tags_and_scores;
  tags_and_scores.reserve(tokens.size());

  for (const auto& sub_vector : tokens) {
    PerTokenListPredictions predictions(sub_vector.size());
    tags_and_scores.push_back(predictions);
  }
  data::ColumnMap data(data::ColumnMap(
      {{_tokens_column, data::ArrayColumn<std::string>::make(std::move(tokens),
                                                             std::nullopt)}}));

  // featurize input data
  auto columns = _inference_transforms->applyStateless(data);
  auto tensors = data::toTensorBatches(columns, _bolt_inputs, 2048);

  size_t sub_vector_index = 0;
  size_t token_index = 0;

  for (const auto& batch : tensors) {
    auto outputs = _bolt_model->forward(batch).at(0);

    for (size_t i = 0; i < outputs->batchSize(); i += 1) {
      if (token_index >= tags_and_scores[sub_vector_index].size()) {
        token_index = 0;
        sub_vector_index++;
      }
      auto token_level_predictions =
          outputs->getVector(i).topKNeurons(top_k + 1);
      while (!token_level_predictions.empty()) {
        float score = token_level_predictions.top().first;
        uint32_t tag = token_level_predictions.top().second;
        tags_and_scores[sub_vector_index][token_index].push_back(
            {label_to_tag_map.at(tag), score});
        token_level_predictions.pop();
      }
      applyPunctAndStopWordFilter(
          data.getArrayColumn<std::string>(_tokens_column)
              ->row(sub_vector_index)[token_index],
          tags_and_scores[sub_vector_index][token_index],
          label_to_tag_map.at(0));

      bool removed_highest = false;

      auto highest_tag_act =
          tags_and_scores[sub_vector_index][token_index].back();

      auto second_highest_tag_act =
          tags_and_scores[sub_vector_index][token_index][top_k - 1];

      if (tag_to_label_map.at(highest_tag_act.first) == 0 &&
          highest_tag_act.second < 0.9 &&
          second_highest_tag_act.second >= 0.05) {
        tags_and_scores[sub_vector_index][token_index].pop_back();
        removed_highest = true;
      }

      // topkactivation is a min heap hence, reverse it
      std::reverse(tags_and_scores[sub_vector_index][token_index].begin(),
                   tags_and_scores[sub_vector_index][token_index].end());

      if (!removed_highest) {
        tags_and_scores[sub_vector_index][token_index].pop_back();
      }

      if (sub_vector_index >= tags_and_scores.size()) {
        throw std::runtime_error("tags indices not matching");
      }
      token_index += 1;
    }
  }
  return tags_and_scores;
}

}  // namespace thirdai::bolt::NER
