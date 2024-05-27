#pragma once

#include <wrappers/src/EigenDenseWrapper.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
#include <stdexcept>

namespace thirdai::bolt {

/**
 * The class is for optimized dense inference for models that have an embedding
 * layer followed by a fully connected layer. It uses a matrix matrix
 * multiplication funciton that is faster than regular bolt for dense inference.
 * It cannot be integrated directly into bolt because the parallelism structure
 * forces bolt to use gemv instead of gemm.
 */
class EmbFcInference {
 public:
  EmbFcInference(EmbeddingPtr emb, const FullyConnectedPtr& fc);

  TensorPtr forward(const TensorPtr& input);

 private:
  using Matrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  EmbeddingPtr _emb;
  FullyConnectedPtr _fc;

  Eigen::Map<Matrix> _weights;
  Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> _biases;
};

}  // namespace thirdai::bolt