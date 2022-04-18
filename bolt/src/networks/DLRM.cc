#include "DLRM.h"
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/ProgressBar.h>
#include <atomic>
#include <chrono>
#include <stdexcept>

namespace thirdai::bolt {

DLRM::DLRM(
    EmbeddingLayerConfig embedding_config,
    std::vector<std::shared_ptr<SequentialLayerConfig>> bottom_mlp_configs,
    std::vector<std::shared_ptr<SequentialLayerConfig>> top_mlp_configs,
    uint32_t dense_feature_dim)
    : _embedding_layer(embedding_config),
      _bottom_mlp(bottom_mlp_configs, dense_feature_dim),
      _top_mlp(top_mlp_configs, (embedding_config.lookup_size *
                                 embedding_config.num_embedding_lookups) +
                                    bottom_mlp_configs.back()->getDim()),

      _iter(0),
      _epoch_count(0) {
  if (bottom_mlp_configs.back()->getSparsity() != 1.0) {
    throw std::invalid_argument("Dense feature layer must have sparsity 1.0");
  }

  if (top_mlp_configs.back()->getDim() == 1 &&
      top_mlp_configs.back()->getActFunc() != ActivationFunction::Linear) {
    throw std::invalid_argument(
        "Output layer must have MeanSqauredError if dimension is 1");
  }

  if (top_mlp_configs.back()->getDim() != 1 &&
      top_mlp_configs.back()->getActFunc() != ActivationFunction::Softmax) {
    throw std::invalid_argument(
        "Output layer must have Softmax if dimension is > 1");
  }

  _softmax =
      top_mlp_configs.back()->getActFunc() == ActivationFunction::Softmax;
  _output_dim = top_mlp_configs.back()->getDim();

  _concat_layer_dim =
      _embedding_layer.getEmbeddingDim() + bottom_mlp_configs.back()->getDim();
}

void DLRM::initializeNetworkState(uint32_t batch_size, bool force_dense) {
  _concat_layer_state = BoltBatch(_concat_layer_dim, batch_size, true);

  uint32_t embedding_dim = _embedding_layer.getEmbeddingDim();
  uint32_t bottom_mlp_output_dim = _bottom_mlp.outputDim();

  _bottom_mlp_output.clear();
  _embedding_layer_output.clear();

  _bottom_mlp_output.reserve(batch_size);
  _embedding_layer_output.reserve(batch_size);

  for (uint32_t b = 0; b < batch_size; b++) {
    const BoltVector& concat_vec = _concat_layer_state[b];
    _bottom_mlp_output.emplace_back(nullptr, concat_vec.activations,
                                    concat_vec.gradients,
                                    bottom_mlp_output_dim);

    _embedding_layer_output.emplace_back(
        nullptr, concat_vec.activations + bottom_mlp_output_dim,
        concat_vec.gradients + bottom_mlp_output_dim, embedding_dim);
  }

  _embedding_layer.initializeLayer(batch_size);

  _bottom_mlp.initializeNetworkState(batch_size, force_dense);
  _top_mlp.initializeNetworkState(batch_size, force_dense);
}

void DLRM::forward(uint32_t batch_index,
                   const dataset::ClickThroughBatch& inputs, BoltVector& output,
                   bool train) {
  (void)train;
  _bottom_mlp.forward(batch_index, inputs[batch_index],
                      _bottom_mlp_output[batch_index], nullptr);

  _embedding_layer.forward(batch_index, inputs.categoricalFeatures(batch_index),
                           _embedding_layer_output[batch_index]);

  _top_mlp.forward(batch_index, _concat_layer_state[batch_index], output,
                   nullptr);
}

void DLRM::backpropagate(uint32_t batch_index,
                         dataset::ClickThroughBatch& inputs,
                         BoltVector& output) {
  _top_mlp.backpropagate<false>(batch_index, _concat_layer_state[batch_index],
                                output);

  _embedding_layer.backpropagate(batch_index,
                                 _embedding_layer_output[batch_index]);

  _bottom_mlp.backpropagate<true>(batch_index, inputs[batch_index],
                                  _bottom_mlp_output[batch_index]);
}

}  // namespace thirdai::bolt