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
#include <string>

namespace thirdai::bolt {

DyadicModel::DyadicModel(bolt::ModelPtr model, bool is_bidirectional)
    : _model(std::move(model)) {
  size_t model_inputs = _model->inputs().size();

  assert(!is_bidirectional || (model_inputs % 2 == 0));

  size_t n_intervals = is_bidirectional ? model_inputs / 2 : model_inputs;

  _dyadic_transform = std::make_shared<data::DyadicInterval>(
      "target", std::nullopt, "interval_", "next_word", std::nullopt, n_intervals);

  if (_model->outputs().size() != 1) {
    throw std::invalid_argument("Expected model to have a single output.");
  }

  _vocab_size = _model->outputs().at(0)->dim();

  for (size_t i = 0; i < n_intervals; i++) {
    _bolt_inputs.push_back(
        data::OutputColumns("interval_from_end_" + std::to_string(1 << i)));
  }
  if (is_bidirectional) {
    for (size_t i = 0; i < n_intervals; i++) {
      _bolt_inputs.push_back(
          data::OutputColumns("interval_from_start_" + std::to_string(1 << i)));
    }
  }
}

bolt::TensorPtr DyadicModel::nextTokenProbs(
    std::vector<uint32_t>& prompt, std::vector<std::vector<uint32_t>> tokens) {
  (void)prompt;
  data::ColumnMap data({{"target", data::ArrayColumn<uint32_t>::make(
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
      data::Pipeline::make({std::make_shared<data::StringToTokenArray>(
                                "target", "target", ' ', _vocab_size),
                            _dyadic_transform});

  return data::Loader(data_iter, transform, nullptr, _bolt_inputs,
                      {data::OutputColumns("next_word")},
                      /* batch_size= */ batch_size,
                      /* shuffle= */ shuffle, /* verbose= */ true,
                      /* shuffle_buffer_size= */ 200000);
}

}  // namespace thirdai::bolt