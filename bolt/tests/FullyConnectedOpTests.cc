#include <wrappers/src/EigenDenseWrapper.h>
#include "gtest/gtest.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <Eigen/src/Core/util/Constants.h>
#include <algorithm>
#include <random>
#include <unordered_set>

namespace thirdai::bolt::nn::tests {

namespace {

std::mt19937 rnd(4092);

}  // namespace

std::vector<uint32_t> randomIndices(uint32_t n_vecs, uint32_t dim,
                                    uint32_t nonzeros) {
  std::vector<uint32_t> indices;
  indices.reserve(n_vecs * nonzeros);

  std::uniform_int_distribution<uint32_t> dist(0, dim - 1);

  for (uint32_t i = 0; i < n_vecs; i++) {
    std::unordered_set<uint32_t> vec_indices;
    while (vec_indices.size() < nonzeros) {
      vec_indices.insert(dist(rnd));
    }

    indices.insert(indices.end(), vec_indices.begin(), vec_indices.end());
  }

  return indices;
}

std::vector<float> randomValues(uint32_t n_vecs, uint32_t dim) {
  std::vector<float> values(n_vecs * dim);

  std::uniform_int_distribution<int> dist(-40, 40);

  std::generate(values.begin(), values.end(),
                [&dist]() { return 0.125 * dist(rnd); });

  return values;
}

std::vector<float> sparseToDense(uint32_t n_vecs, uint32_t dim,
                                 uint32_t nonzeros,
                                 const std::vector<uint32_t>& indices,
                                 const std::vector<float>& values) {
  std::vector<float> dense_values(n_vecs * dim, 0.0);

  for (uint32_t i = 0; i < n_vecs; i++) {
    for (uint32_t j = 0; j < nonzeros; j++) {
      dense_values.at(i * dim + indices[i * nonzeros + j]) =
          values.at(i * nonzeros + j);
    }
  }

  return dense_values;
}

void setWeightsAndBiases(ops::FullyConnectedPtr& op) {
  std::vector<float> weights = randomValues(op->dim(), op->inputDim());
  std::vector<float> biases = randomValues(op->dim(), 1);

  op->setWeights(weights.data());
  op->setBiases(biases.data());
}

void setGradients(tensor::TensorPtr& tensor, const std::vector<float>& grads) {
  ASSERT_EQ(tensor->gradients().size(), grads.size());
  std::copy(grads.begin(), grads.end(), tensor->gradients().begin());
}

using EigenMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenVector = Eigen::VectorXf;

EigenMatrix weightsToEigen(const ops::FullyConnectedPtr& op) {
  EigenMatrix weights(op->dim(), op->inputDim());
  std::copy(op->weightsPtr(), op->weightsPtr() + op->inputDim() * op->dim(),
            weights.data());

  return weights;
}

EigenVector biasesToEigen(const ops::FullyConnectedPtr& op) {
  EigenVector biases(op->dim());
  std::copy(op->biasesPtr(), op->biasesPtr() + op->dim(), biases.data());

  return biases;
}

EigenMatrix dataToEigen(uint32_t n_vecs, uint32_t dim,
                        const std::vector<float>& values) {
  EXPECT_EQ(values.size(), n_vecs * dim);

  EigenMatrix tensor(n_vecs, dim);

  std::copy(values.begin(), values.end(), tensor.data());

  return tensor;
}

EigenMatrix dataToEigen(uint32_t n_vecs, uint32_t dim, uint32_t nonzeros,
                        const std::vector<uint32_t>& indices,
                        const std::vector<float>& values) {
  auto dense_values = sparseToDense(n_vecs, dim, nonzeros, indices, values);

  return dataToEigen(n_vecs, dim, dense_values);
}

tensor::TensorPtr dataToTensor(std::vector<uint32_t> dims,
                               const std::vector<float>& values) {
  return tensor::Tensor::fromArray(nullptr, values.data(), dims, dims.back(),
                                   /* with_grad= */ true);
}

tensor::TensorPtr dataToTensor(std::vector<uint32_t> dims, uint32_t nonzeros,
                               const std::vector<uint32_t>& indices,
                               const std::vector<float>& values) {
  return tensor::Tensor::fromArray(indices.data(), values.data(),
                                   std::move(dims), nonzeros,
                                   /* with_grad= */ true);
}

EigenMatrix eigenForward(const EigenMatrix& input, const EigenMatrix& weights,
                         const EigenVector& biases) {
  EigenMatrix out = input * weights.transpose();

  out.rowwise() += biases.transpose();

  return out;
}

EigenMatrix eigenBackpropagateInputGrads(const EigenMatrix& output_grads,
                                         const EigenMatrix& weights) {
  return output_grads * weights;
}

EigenMatrix eigenBackpropagateWeightGrads(const EigenMatrix& inputs,
                                          const EigenMatrix& output_grads) {
  return output_grads.transpose() * inputs;
}

void runForward(autograd::ComputationPtr& comp, uint32_t batch_size) {
  for (uint32_t i = 0; i < batch_size; i++) {
    comp->forward(i, /* training= */ true);
  }
}

void runBackpropagate(autograd::ComputationPtr& comp, uint32_t batch_size) {
  for (uint32_t i = 0; i < batch_size; i++) {
    comp->backpropagate(i);
  }
}

uint32_t BATCH_SIZE = 4, INNER_DIM_1 = 6, INNER_DIM_2 = 8, INPUT_DIM = 10,
         OUTPUT_DIM = 20, INPUT_NONZEROS = 5, OUTPUT_NONZEROS = 10;

uint32_t N_VECS = BATCH_SIZE * INNER_DIM_1 * INNER_DIM_2;

std::vector<uint32_t> INPUT_TENSOR_DIMS = {BATCH_SIZE, INNER_DIM_1, INNER_DIM_2,
                                           INPUT_DIM};

std::vector<uint32_t> OUTPUT_TENSOR_DIMS = {BATCH_SIZE, INNER_DIM_1,
                                            INNER_DIM_2, OUTPUT_DIM};

std::tuple<autograd::ComputationPtr, ops::FullyConnectedPtr,
           autograd::ComputationPtr>
makeOp(float sparsity) {
  auto input = ops::Input::make({INNER_DIM_1, INNER_DIM_2, INPUT_DIM});

  auto op = ops::FullyConnected::make(/* dim= */ OUTPUT_DIM,
                                      /* input_dim= */ INPUT_DIM,
                                      /* sparsity= */ sparsity,
                                      /* activation= */ "linear");
  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->allocate(BATCH_SIZE, /* use_sparsity= */ true);

  return {input, op, output};
}

void checkDenseTensor(const EigenMatrix& eigen, const tensor::TensorPtr& bolt,
                      bool check_grads) {
  ASSERT_EQ(eigen.cols(), bolt->dims().back());
  ASSERT_EQ(eigen.rows(), N_VECS);
  ASSERT_FALSE(bolt->isSparse());
  ASSERT_TRUE(bolt->nonzeros().has_value());
  ASSERT_EQ(bolt->nonzeros().value(), bolt->dims().back());

  std::vector<float> bolt_arr = bolt->values();
  if (check_grads) {
    bolt_arr = bolt->gradients();
  }

  ASSERT_EQ(bolt_arr.size(), N_VECS * bolt->dims().back());

  for (uint32_t i = 0; i < N_VECS * bolt->dims().back(); i++) {
    ASSERT_FLOAT_EQ(eigen.data()[i], bolt_arr.at(i));
  }
}

void checkSparseTensor(const EigenMatrix& eigen, const tensor::TensorPtr& bolt,
                       bool check_grads) {
  ASSERT_EQ(eigen.cols(), bolt->dims().back());
  ASSERT_EQ(eigen.rows(), N_VECS);
  ASSERT_TRUE(bolt->isSparse());
  ASSERT_TRUE(bolt->nonzeros().has_value());

  uint32_t nonzeros = bolt->nonzeros().value();

  std::vector<float> bolt_arr = bolt->values();
  if (check_grads) {
    bolt_arr = bolt->gradients();
  }

  ASSERT_EQ(bolt_arr.size(), N_VECS * nonzeros);

  for (uint32_t i = 0; i < N_VECS; i++) {
    for (uint32_t j = 0; j < nonzeros; j++) {
      uint32_t index = bolt->indices().at(i * nonzeros + j);

      ASSERT_FLOAT_EQ(eigen(i, index), bolt_arr.at(i * nonzeros + j));
    }
  }
}

void checkWeightGrads(const EigenMatrix& eigen_weight_grads,
                      const ops::FullyConnectedPtr& op) {
  ASSERT_EQ(eigen_weight_grads.rows(), op->dim());
  ASSERT_EQ(eigen_weight_grads.cols(), op->inputDim());

  const float* weight_grads = op->gradients().at(0)->data();
  for (uint32_t i = 0; i < op->inputDim() * op->dim(); i++) {
    ASSERT_FLOAT_EQ(eigen_weight_grads.data()[i], weight_grads[i]);
  }
}

void runTest(const EigenMatrix& eigen_input,
             const tensor::TensorPtr& bolt_input,
             const EigenMatrix& eigen_output_grad,
             const tensor::TensorPtr& bolt_output_grad) {
  auto [input, op, output] = makeOp(bolt_output_grad->isSparse() ? 0.5 : 1.0);

  auto eigen_weights = weightsToEigen(op);

  input->setTensor(bolt_input);

  if (bolt_output_grad->isSparse()) {
    auto labels = ops::Input::make({INNER_DIM_1, INNER_DIM_2, OUTPUT_DIM});

    output->addInput(labels);

    labels->setTensor(bolt_output_grad);
  }

  runForward(output, BATCH_SIZE);

  auto eigen_output =
      eigenForward(eigen_input, eigen_weights, biasesToEigen(op));

  if (bolt_output_grad->isSparse()) {
    checkSparseTensor(eigen_output, output->tensor(), /* check_grads= */ false);
  } else {
    checkDenseTensor(eigen_output, output->tensor(), /* check_grads= */ false);
  }

  setGradients(output->tensor(), bolt_output_grad->values());

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_input_grad =
      eigenBackpropagateInputGrads(eigen_output_grad, eigen_weights);

  if (bolt_input->isSparse()) {
    checkSparseTensor(eigen_input_grad, input->tensor(),
                      /* check_grads= */ true);
  } else {
    checkDenseTensor(eigen_input_grad, input->tensor(),
                     /* check_grads= */ true);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_input, eigen_output_grad);

  checkWeightGrads(eigen_weight_grads, op);
}

std::pair<EigenMatrix, tensor::TensorPtr> denseData(uint32_t dim) {
  auto values = randomValues(N_VECS, dim);

  auto eigen = dataToEigen(N_VECS, dim, values);
  auto bolt = dataToTensor({BATCH_SIZE, INNER_DIM_1, INNER_DIM_2, dim}, values);

  return {eigen, bolt};
}

std::pair<EigenMatrix, tensor::TensorPtr> sparseData(uint32_t dim,
                                                     uint32_t nonzeros) {
  auto indices = randomIndices(N_VECS, dim, nonzeros);
  auto values = randomValues(N_VECS, nonzeros);

  auto eigen = dataToEigen(N_VECS, dim, nonzeros, indices, values);
  auto bolt = dataToTensor({BATCH_SIZE, INNER_DIM_1, INNER_DIM_2, dim},
                           nonzeros, indices, values);

  return {eigen, bolt};
}

TEST(FullyConnectedOpTests, DenseDense) {
  auto [eigen_input, bolt_input] = denseData(INPUT_DIM);

  auto [eigen_output_grad, bolt_output_grad] = denseData(OUTPUT_DIM);

  ASSERT_FALSE(bolt_input->isSparse());
  ASSERT_FALSE(bolt_output_grad->isSparse());

  runTest(eigen_input, bolt_input, eigen_output_grad, bolt_output_grad);
}

TEST(FullyConnectedOpTests, SparseDense) {
  auto [eigen_input, bolt_input] = sparseData(INPUT_DIM, INPUT_NONZEROS);

  auto [eigen_output_grad, bolt_output_grad] = denseData(OUTPUT_DIM);

  ASSERT_TRUE(bolt_input->isSparse());
  ASSERT_FALSE(bolt_output_grad->isSparse());

  runTest(eigen_input, bolt_input, eigen_output_grad, bolt_output_grad);
}

TEST(FullyConnectedOpTests, DenseSparse) {
  auto [eigen_input, bolt_input] = denseData(INPUT_DIM);

  auto [eigen_output_grad, bolt_output_grad] =
      sparseData(OUTPUT_DIM, OUTPUT_NONZEROS);

  ASSERT_FALSE(bolt_input->isSparse());
  ASSERT_TRUE(bolt_output_grad->isSparse());

  runTest(eigen_input, bolt_input, eigen_output_grad, bolt_output_grad);
}

TEST(FullyConnectedOpTests, SparseSparse) {
  auto [eigen_input, bolt_input] = sparseData(INPUT_DIM, INPUT_NONZEROS);

  auto [eigen_output_grad, bolt_output_grad] =
      sparseData(OUTPUT_DIM, OUTPUT_NONZEROS);

  ASSERT_TRUE(bolt_input->isSparse());
  ASSERT_TRUE(bolt_output_grad->isSparse());

  runTest(eigen_input, bolt_input, eigen_output_grad, bolt_output_grad);
}

}  // namespace thirdai::bolt::nn::tests