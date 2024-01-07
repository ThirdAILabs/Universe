#include "DyadicModel.h"
#include <bolt/src/train/trainer/Dataset.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

DyadicModel::DyadicModel(bolt::ModelPtr model,
                         const data::DyadicInterval& dyadic_transform,
                         data::OutputColumnsList bolt_inputs)
    : _model(std::move(model)),
      _dyadic_transform(
          std::make_shared<data::DyadicInterval>(dyadic_transform)),
      _bolt_inputs(std::move(bolt_inputs)) {
  if (_model->outputs().size() != 1) {
    throw std::invalid_argument("Expected model to have a single output.");
  }

  _vocab_size = _model->outputs().at(0)->dim();
}

std::pair<bolt::TensorPtr, std::vector<std::vector<size_t>>>
DyadicModel::nextTokenProbs(
    const std::vector<std::vector<uint32_t>>& prompts,
    const std::vector<std::vector<std::vector<uint32_t>>>& tokens) {
  auto prompt_column_name = _dyadic_transform->getPromptColumn();
  // each of the chunks inside tokens should be of the same length
  size_t tokens_size = tokens[0].size();
  std::vector<std::vector<uint32_t>> flattened_tokens;

  // stores a mapping where mapping[i][j] stores location of token[i][j] in
  // flttened_tokens
  std::vector<std::vector<size_t>> mapping;

  size_t current_location = 0;
  for (const auto& batch : tokens) {
    std::vector<size_t> local_map;
    for (const auto& tokens : batch) {
      flattened_tokens.push_back(tokens);
      local_map.push_back(current_location);
      current_location += 1;
    }
    mapping.push_back(local_map);
  }

  data::ColumnMap data(
      data::ColumnMap({{_dyadic_transform->getInputColumn(),
                        data::ArrayColumn<uint32_t>::make(
                            std::move(flattened_tokens), _vocab_size)}}));

  if (prompt_column_name) {
    std::vector<std::vector<uint32_t>> flatten_prompts;
    for (const auto& prompt : prompts) {
      std::vector<std::vector<uint32_t>> prompt_column(tokens_size, prompt);
      flatten_prompts.insert(flatten_prompts.end(), prompt_column.begin(),
                             prompt_column.end());
    }
    data.setColumn(*prompt_column_name,
                   data::ArrayColumn<uint32_t>::make(std::move(flatten_prompts),
                                                     _vocab_size));
  }

  auto columns = _dyadic_transform->inferenceFeaturization(data);

  auto tensors = data::toTensors(columns, _bolt_inputs);

  auto token_probs = _model->forward(tensors).at(0);

  return {token_probs, mapping};
}

metrics::History DyadicModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    std::optional<size_t> max_in_memory_batches,
    const DistributedCommPtr& comm) {
  size_t batches_to_load =
      max_in_memory_batches.value_or(data::Loader::NO_LIMIT);

  auto train_dataset_loader =
      getDataLoader(train_data, batch_size, /* shuffle= */ true);
  auto val_dataset =
      getDataLoader(val_data, batch_size, /* shuffle= */ false).all();

  Trainer trainer(_model);

  // We cannot use train_with_dataset_loader, since it is using the older
  // dataset::DatasetLoader while dyadic model is using data::Loader
  for (uint32_t e = 0; e < epochs; e++) {
    while (auto train_chunk = train_dataset_loader.next(batches_to_load)) {
      if (train_chunk) {
        trainer.train_with_metric_names(
            *train_chunk, learning_rate, 1, train_metrics, val_dataset,
            val_metrics, /* steps_per_validation= */ std::nullopt,
            /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
            /* autotune_rehash_rebuild= */ false, /* verbose= */ true,
            /* logging_interval= */ std::nullopt, comm);
      }
    }
  }
  return trainer.getHistory();
}

data::Loader DyadicModel::getDataLoader(const dataset::DataSourcePtr& data,
                                        size_t batch_size, bool shuffle) {
  std::vector<std::string> columns_names = {
      _dyadic_transform->getInputColumn()};
  auto prompt_column = _dyadic_transform->getPromptColumn();
  auto context_column = _dyadic_transform->getContextColumn();
  if (prompt_column) {
    columns_names.push_back(*prompt_column);
  }
  if (context_column) {
    columns_names.push_back(*context_column);
  }

  auto data_iter = data::JsonIterator::make(data, columns_names, 1000);
  auto transform =
      data::Pipeline::make({std::make_shared<data::StringToTokenArray>(
          _dyadic_transform->getInputColumn(),
          _dyadic_transform->getInputColumn(), ' ', _vocab_size)});
  if (prompt_column) {
    transform = transform->then(std::make_shared<data::StringToTokenArray>(
        *prompt_column, *prompt_column, ' ', _vocab_size));
  }
  if (context_column) {
    transform = transform->then(std::make_shared<data::StringToTokenArray>(
        *context_column, *context_column, ' ', _vocab_size));
  }
  transform = transform->then(_dyadic_transform);
  return data::Loader(
      data_iter, transform, nullptr, _bolt_inputs,
      {data::OutputColumns(_dyadic_transform->getTargetColumn())},
      /* batch_size= */ batch_size,
      /* shuffle= */ shuffle, /* verbose= */ true,
      /* shuffle_buffer_size= */ 200000);
}

}  // namespace thirdai::bolt