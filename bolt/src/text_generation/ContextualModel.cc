#include "ContextualModel.h"

namespace thirdai::bolt {

ContextualModel::ContextualModel(
    bolt::ModelPtr model, dataset::TextGenerationFeaturizerPtr featurizer)
    : _model(std::move(model)), _featurizer(std::move(featurizer)) {}

bolt::TensorPtr ContextualModel::nextTokenProbs(
    std::vector<std::vector<uint32_t>> tokens) {
  auto tensors = _featurizer->featurizeInputBatch(tokens);
  return _model->forward(tensors).at(0);
}

}  // namespace thirdai::bolt