#include "EmbeddingInference.h"
#include <stdexcept>

namespace thirdai::bolt {

EmbeddingInference::EmbeddingInference(EmbeddingPtr emb,
                                       const FullyConnectedPtr& fc)
    : _emb(std::move(emb)),
      _fc(fc),
      _weights(const_cast<float*>(fc->weightsPtr()), fc->dim(), fc->inputDim()),
      _biases(const_cast<float*>(fc->biasesPtr()), 1, fc->dim()) {
  if (_emb->dim() != fc->inputDim()) {
    throw std::invalid_argument(
        "Embedding dim must match input dim to classification head.");
  }
  if (fc->kernel()->getActivationFunction() != ActivationFunction::Sigmoid) {
    throw std::invalid_argument("Unsupported output activation function.");
  }
}

TensorPtr EmbeddingInference::forward(const TensorPtr& input) {
  Matrix embs(input->batchSize(), _emb->dim());

#pragma omp parallel for default(none) shared(input, embs)
  for (size_t i = 0; i < input->batchSize(); i++) {
    _emb->forward(input->getVector(i), embs.row(i).data());
  }

  auto output = Tensor::dense(input->batchSize(), _fc->dim());

  Eigen::Map<Matrix> out(const_cast<float*>(output->activationsPtr()),
                         input->batchSize(), _fc->dim());

  out.noalias() = (embs * _weights.transpose());
  out.rowwise() += _biases;

  out.array() = (1 + (-out.array()).exp()).inverse();

  return output;
}

}  // namespace thirdai::bolt