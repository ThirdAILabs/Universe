#include "DyadicModel.h"
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/DyadicInterval.h>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

DyadicModel::DyadicModel(bolt::ModelPtr model) : _model(std::move(model)) {
  size_t n_intervals = _model->inputs().size();

  _dyadic_transform = std::make_shared<data::DyadicInterval>(
      "target", "interval_", "next_word", n_intervals);

  if (_model->outputs().size() != 1) {
    throw std::invalid_argument("Expected model to have a single output.");
  }

  _vocab_size = _model->outputs().at(0)->dim();

  for (size_t i = 0; i < n_intervals; i++) {
    _bolt_inputs.push_back(
        data::OutputColumns("interval_" + std::to_string(1 << i)));
  }
}

bolt::TensorPtr DyadicModel::nextTokenProbs(
    std::vector<std::vector<uint32_t>> tokens) {
  data::ColumnMap data({{"target", data::ArrayColumn<uint32_t>::make(
                                       std::move(tokens), _vocab_size)}});

  auto intervals = _dyadic_transform->inferenceFeaturization(data);

  auto tensors = data::toTensors(intervals, _bolt_inputs);

  return _model->forward(tensors).at(0);
}

}  // namespace thirdai::bolt