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

bolt::TensorPtr DyadicModel::nextTokenProbs(
    std::vector<uint32_t>& prompts, std::vector<std::vector<uint32_t>> tokens) {
  auto prompt_column_name = _dyadic_transform->getPromptColumn();
  size_t tokens_size = tokens.size();
  data::ColumnMap data(data::ColumnMap(
      {{_dyadic_transform->getInputColumn(),
        data::ArrayColumn<uint32_t>::make(std::move(tokens), _vocab_size)}}));

<<<<<<< HEAD
          : data::ColumnMap({{"target", data::ArrayColumn<uint32_t>::make(
                                            std::move(tokens), _vocab_size)},
                             {"context", data::ArrayColumn<uint32_t>::make(
                                             std::move(tokens), _vocab_size)}});
=======
  if (prompt_column_name) {
    std::vector<std::vector<uint32_t>> prompt_columns(tokens_size, prompts);
    data.setColumn(*prompt_column_name,
                   data::ArrayColumn<uint32_t>::make(std::move(prompt_columns),
                                                     _vocab_size));
  }
>>>>>>> origin/add-options-to-dyadic-model

          auto columns = _dyadic_transform->inferenceFeaturization(data);

          auto tensors = data::toTensors(columns, _bolt_inputs);

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
  std::vector<std::string> columns_names = {
      _dyadic_transform->getInputColumn()};
  auto prompt_column = _dyadic_transform->getPromptColumn();
  if (prompt_column) {
    columns_names.push_back(*prompt_column);
  }
  auto context_column = _dyadic_transform->getContextColumn();

  auto data_iter = data::JsonIterator::make(data, columns_names);
  auto transform = data::Pipeline::make(
      {std::make_shared<data::StringToTokenArray>(
           _dyadic_transform->getInputColumn(),
           _dyadic_transform->getInputColumn(), ' ', _vocab_size),
       _dyadic_transform});
  if (prompt_column) {
    transform->then(std::make_shared<data::StringToTokenArray>(
        *prompt_column, *prompt_column, ' ', _vocab_size));
  }
  if (context_column) {
    transform->then(std::make_shared<data::StringToTokenArray>(
        *context_column, *context_column, ' ', _vocab_size));
  }
  return data::Loader(
      data_iter, transform, nullptr, _bolt_inputs,
      {data::OutputColumns(_dyadic_transform->getTargetColumn())},
      /* batch_size= */ batch_size,
      /* shuffle= */ shuffle, /* verbose= */ true,
      /* shuffle_buffer_size= */ 200000);
}

}  // namespace thirdai::bolt