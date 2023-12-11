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
#include <optional>
#include <stdexcept>

namespace thirdai::bolt {

DyadicModel::DyadicModel(bolt::ModelPtr model,
                         data::DyadicInterval dyadic_transform,
                         data::OutputColumnsList bolt_inputs,
                         bool is_prompt_needed)
    : _model(std::move(model)),
      _dyadic_transform(
          std::make_shared<data::DyadicInterval>(dyadic_transform)),
      _bolt_inputs(bolt_inputs),
      _is_prompt_needed(is_prompt_needed) {
  if (_model->outputs().size() != 1) {
    throw std::invalid_argument("Expected model to have a single output.");
  }

  _vocab_size = _model->outputs().at(0)->dim();
}

bolt::TensorPtr DyadicModel::nextTokenProbs(
    std::vector<uint32_t>& prompts, std::vector<std::vector<uint32_t>> tokens) {
  // TODO(pratik):Handle for multiple columns
  data::ColumnMap data =
      _is_prompt_needed
          ? data::ColumnMap({{"target", data::ArrayColumn<uint32_t>::make(
                                            std::move(tokens), _vocab_size)},
                             {"prompt", data::ArrayColumn<uint32_t>::make(
                                            {prompts}, _vocab_size)}})

          : data::ColumnMap({{"target", data::ArrayColumn<uint32_t>::make(
                                            std::move(tokens), _vocab_size)}});

  auto intervals = _dyadic_transform->inferenceFeaturization(data);

  auto tensors = data::toTensors(intervals, _bolt_inputs);

  return _model->forward(tensors).at(0);
}

metrics::History DyadicModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const DistributedCommPtr& comm) {
  auto train_dataset =
      getDataLoader(train_data, batch_size, /* shuffle= */ true).all();
  auto val_dataset =
      getDataLoader(val_data, batch_size, /* shuffle= */ false).all();

  Trainer trainer(_model);

  return trainer.train_with_metric_names(
      train_dataset, learning_rate, epochs, train_metrics, val_dataset,
      val_metrics, /* steps_per_validation= */ std::nullopt,
      /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ false, /* verbose= */ true,
      /* logging_interval= */ std::nullopt, comm);
}

data::Loader DyadicModel::getDataLoader(const dataset::DataSourcePtr& data,
                                        size_t batch_size, bool shuffle) {
  auto data_iter = data::JsonIterator::make(data, {"target"});
  auto transform =
      _is_prompt_needed
          ? data::Pipeline::make({std::make_shared<data::StringToTokenArray>(
                                      "target", "target", ' ', _vocab_size),
                                  _dyadic_transform,
                                  std::make_shared<data::StringToTokenArray>(
                                      "prompt", "prompt", ' ', _vocab_size)})

          : data::Pipeline::make({std::make_shared<data::StringToTokenArray>(
                                      "target", "target", ' ', _vocab_size),
                                  _dyadic_transform});

  return data::Loader(data_iter, transform, nullptr, _bolt_inputs,
                      {data::OutputColumns("next_word")},
                      /* batch_size= */ batch_size,
                      /* shuffle= */ shuffle, /* verbose= */ true,
                      /* shuffle_buffer_size= */ 200000);
}

}  // namespace thirdai::bolt