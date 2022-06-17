#include "TextEmbeddingModel.h"

namespace thirdai::bolt {

void TextEmbeddingModel::forward(uint32_t batch_index,
                                 const dataset::MaskedSentenceBatch& input,
                                 BoltVector& output, const BoltVector* labels) {
  _sentence_embedding_model.forward(batch_index, input[batch_index],
                                    _sentence_embedding_output[batch_index],
                                    nullptr);

  _token_embedding_layers_used.at(input.maskedIndex(batch_index)) = true;
  _token_embedding_layers.at(input.maskedIndex(batch_index))
      .forward(_sentence_embedding_output[batch_index],
               _token_embedding_output[batch_index], nullptr);

  _classifier.forward(batch_index, _token_embedding_output[batch_index], output,
                      labels);
}

void TextEmbeddingModel::backpropagate(uint32_t batch_index,
                                       dataset::MaskedSentenceBatch& input,
                                       BoltVector& output) {
  _classifier.backpropagate<false>(
      batch_index, _token_embedding_output[batch_index], output);

  _token_embedding_layers.at(input.maskedIndex(batch_index))
      .backpropagate(_sentence_embedding_output[batch_index],
                     _token_embedding_output[batch_index]);

  _sentence_embedding_model.backpropagate<true>(
      batch_index, input[batch_index], _sentence_embedding_output[batch_index]);
}

void TextEmbeddingModel::updateParameters(float learning_rate, uint32_t iter) {
  _sentence_embedding_model.updateParameters(learning_rate, iter);

  for (uint32_t i = 0; i < _token_embedding_layers.size(); i++) {
    if (_token_embedding_layers_used[i]) {
      _token_embedding_layers[i].updateParameters(learning_rate, iter, BETA1,
                                                  BETA2, EPS);
      _token_embedding_layers_used[i] = false;
    }
  }

  _classifier.updateParameters(learning_rate, iter);
}

void TextEmbeddingModel::initializeNetworkState(uint32_t batch_size,
                                                bool force_dense) {
  _sentence_embedding_model.initializeNetworkState(batch_size, force_dense);

  _sentence_embedding_output =
      _sentence_embedding_model.getOutputs(batch_size, force_dense);

  _token_embedding_output =
      _token_embedding_layers[0].createBatchState(batch_size, force_dense);

  _classifier.initializeNetworkState(batch_size, force_dense);
}

}  // namespace thirdai::bolt