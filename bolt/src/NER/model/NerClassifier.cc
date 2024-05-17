#include "NerClassifier.h"
#include <cstdint>
#include <unordered_map>

namespace thirdai::bolt::NER {
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
  for (uint32_t e = 0; e < epochs; e++) {
    trainer.train_with_metric_names(
        train_dataset, learning_rate, 1, train_metrics, val_dataset,
        val_metrics, /* steps_per_validation= */ std::nullopt,
        /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
        /* autotune_rehash_rebuild= */ false, /* verbose= */ true);
  }
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
    const std::unordered_map<uint32_t, std::string>& label_to_tag_map) const {
  return thirdai::bolt::NER::getTags(label_to_tag_map, std::move(tokens), top_k,
                                     _tokens_column, _inference_transforms,
                                     _bolt_inputs, _bolt_model);
}

}  // namespace thirdai::bolt::NER