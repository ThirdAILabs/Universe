#include <wrappers/src/EigenDenseWrapper.h>
#include "gtest/gtest.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <algorithm>
#include <cstddef>
#include <random>
#include <unordered_set>

namespace thirdai::bolt::nn::tests {

namespace {

std::mt19937 rnd(4092);

}  // namespace

std::vector<uint32_t> randomIndices(uint32_t batch_size, uint32_t dim1,
                                    uint32_t dim2, uint32_t dim3,
                                    uint32_t nonzeros) {
  uint32_t num_vecs = batch_size * dim1 * dim2;

  std::vector<uint32_t> indices;
  indices.reserve(num_vecs * nonzeros);

  std::uniform_int_distribution<uint32_t> dist(0, dim3 - 1);

  for (uint32_t i = 0; i < num_vecs; i++) {
    std::unordered_set<uint32_t> vec_indices;
    while (vec_indices.size() < nonzeros) {
      vec_indices.insert(dist(rnd));
    }

    indices.insert(indices.end(), vec_indices.begin(), vec_indices.end());
  }

  return indices;
}

std::vector<float> randomValues(uint32_t size) {
  std::vector<float> values(size);

  std::uniform_int_distribution<int> dist(-40, 40);

  std::generate(values.begin(), values.end(),
                [&dist]() { return 0.125 * dist(rnd); });

  return values;
}

std::vector<float> sparseToDense(uint32_t batch_size, uint32_t dim1,
                                 uint32_t dim2, uint32_t dim3,
                                 uint32_t nonzeros,
                                 const std::vector<uint32_t>& indices,
                                 const std::vector<float>& values) {
  uint32_t num_vecs = batch_size * dim1 * dim2;

  std::vector<float> dense_values(num_vecs * dim3, 0.0);

  for (uint32_t i = 0; i < num_vecs; i++) {
    for (uint32_t j = 0; j < nonzeros; j++) {
      dense_values.at(i * dim3 + indices[i * nonzeros + j]) =
          values.at(i * nonzeros + j);
    }
  }

  return dense_values;
}

void setWeightsAndBiases(ops::FullyConnectedPtr& op) {
  std::vector<float> weights = randomValues(op->inputDim() * op->dim());
  std::vector<float> biases = randomValues(op->dim());

  op->setWeightsAndBiases(weights.data(), biases.data());
}

void setGradients(tensor::TensorPtr& tensor, const std::vector<float>& grads) {
  float* grad_ptr = const_cast<float*>(tensor->gradientsPtr());

  std::copy(grads.begin(), grads.end(), grad_ptr);
}

template <int Rank>
using EigenTensor = Eigen::Tensor<float, Rank, Eigen::RowMajor>;
using EigenTensor2D = EigenTensor<2>;
using EigenTensor4D = EigenTensor<4>;

EigenTensor2D weightsToEigen(const ops::FullyConnectedPtr& op) {
  EigenTensor2D weights(op->dim(), op->inputDim());
  std::copy(op->weightsPtr(), op->weightsPtr() + op->inputDim() * op->dim(),
            weights.data());

  return weights;
}

EigenTensor2D biasesToEigen(const ops::FullyConnectedPtr& op) {
  EigenTensor2D biases(1, op->dim());
  std::copy(op->biasesPtr(), op->biasesPtr() + op->dim(), biases.data());

  return biases;
}

EigenTensor4D dataToEigen(uint32_t batch_size, uint32_t dim1, uint32_t dim2,
                          uint32_t dim3, const std::vector<float>& values) {
  EigenTensor4D tensor(batch_size, dim1, dim2, dim3);

  std::copy(values.begin(), values.end(), tensor.data());

  return tensor;
}

EigenTensor4D dataToEigen(uint32_t batch_size, uint32_t dim1, uint32_t dim2,
                          uint32_t dim3, uint32_t nonzeros,
                          const std::vector<uint32_t>& indices,
                          const std::vector<float>& values) {
  auto dense_values =
      sparseToDense(batch_size, dim1, dim2, dim3, nonzeros, indices, values);

  return dataToEigen(batch_size, dim1, dim2, dim3, dense_values);
}

tensor::TensorPtr dataToTensor(uint32_t batch_size, uint32_t dim1,
                               uint32_t dim2, uint32_t dim3,
                               const std::vector<float>& values) {
  return tensor::Tensor::fromArray(nullptr, values.data(),
                                   {batch_size, dim1, dim2, dim3}, dim3,
                                   /* with_grad= */ true);
}

tensor::TensorPtr dataToTensor(uint32_t batch_size, uint32_t dim1,
                               uint32_t dim2, uint32_t dim3, uint32_t nonzeros,
                               const std::vector<uint32_t>& indices,
                               const std::vector<float>& values) {
  return tensor::Tensor::fromArray(indices.data(), values.data(),
                                   {batch_size, dim1, dim2, dim3}, nonzeros,
                                   /* with_grad= */ true);
}

EigenTensor4D eigenForward(const EigenTensor4D& input,
                           const EigenTensor2D& weights,
                           const EigenTensor2D& biases) {
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(3, 1)};

  EigenTensor4D out = input.contract(weights, product_dims);

  const auto& out_dims = out.dimensions();
  int64_t n_rows_in_bcast = 1;
  for (const auto* it = out_dims.begin(); it != out_dims.end() - 1; it++) {
    n_rows_in_bcast *= *it;
  }

  Eigen::array<int64_t, 3> bcast({n_rows_in_bcast, 1});

  out.reshape(Eigen::array<int64_t, 2>({n_rows_in_bcast, out_dims.back()})) +=
      biases.broadcast(bcast);

  return out;
}

EigenTensor4D eigenBackpropagateInputGrads(const EigenTensor4D& output_grads,
                                           const EigenTensor2D& weights) {
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(3, 0)};

  EigenTensor4D out = output_grads.contract(weights, product_dims);

  return out;
}

EigenTensor2D eigenBackpropagateWeightGrads(const EigenTensor4D& inputs,
                                            const EigenTensor4D& output_grads) {
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(0, 0)};

  const auto& out_dims = output_grads.dimensions();
  int64_t n_rows = 1;
  for (const auto* it = out_dims.begin(); it != out_dims.end() - 1; it++) {
    n_rows *= *it;
  }

  auto output_grads_2d =
      output_grads.reshape(Eigen::array<int64_t, 2>({n_rows, out_dims.back()}));

  auto inputs_2d = inputs.reshape(
      Eigen::array<int64_t, 2>({n_rows, inputs.dimensions().back()}));

  EigenTensor2D out = output_grads_2d.contract(inputs_2d, product_dims);

  return out;
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

constexpr uint32_t BATCH_SIZE = 4, DIM1 = 6, DIM2 = 8, DIM3 = 10,
                   OUTPUT_DIM = 20;

TEST(FullyConnectedOpTests, DenseDense) {
  auto input = ops::Input::make({DIM1, DIM2, DIM3});

  auto op =
      ops::FullyConnected::make(/* dim= */ OUTPUT_DIM, /* input_dim= */ DIM3,
                                /* sparsity= */ 1.0,
                                /* activation= */ "linear");
  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->allocate(BATCH_SIZE, /* use_sparsity= */ true);

  auto input_data = randomValues(BATCH_SIZE * DIM1 * DIM2 * DIM3);

  input->setTensor(dataToTensor(BATCH_SIZE, DIM1, DIM2, DIM3, input_data));

  runForward(output, BATCH_SIZE);

  auto eigen_inputs = dataToEigen(BATCH_SIZE, DIM1, DIM2, DIM3, input_data);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < BATCH_SIZE * DIM1 * DIM2 * OUTPUT_DIM; i++) {
    ASSERT_FLOAT_EQ(eigen_output.data()[i],
                    output->tensor()->activationsPtr()[i]);
  }

  auto output_grads = randomValues(BATCH_SIZE * DIM1 * DIM2 * OUTPUT_DIM);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_output_grads =
      dataToEigen(BATCH_SIZE, DIM1, DIM2, OUTPUT_DIM, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < BATCH_SIZE * DIM1 * DIM2 * DIM3; i++) {
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[i],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  const float* weight_grads = op->gradients().at(0)->data();
  for (uint32_t i = 0; i < op->inputDim() * op->dim(); i++) {
    ASSERT_FLOAT_EQ(eigen_weight_grads.data()[i], weight_grads[i]);
  }
}

TEST(FullyConnectedOpTests, SparseDense) {
  auto input = ops::Input::make({DIM1, DIM2, DIM3});

  auto op =
      ops::FullyConnected::make(/* dim= */ OUTPUT_DIM, /* input_dim= */ DIM3,
                                /* sparsity= */ 1.0,
                                /* activation= */ "linear");
  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->allocate(BATCH_SIZE, /* use_sparsity= */ true);

  auto input_indices = randomIndices(BATCH_SIZE, DIM1, DIM2, DIM3, DIM3 / 2);
  auto input_values = randomValues(BATCH_SIZE * DIM1 * DIM2 * DIM3 / 2);

  input->setTensor(dataToTensor(BATCH_SIZE, DIM1, DIM2, DIM3, DIM3 / 2,
                                input_indices, input_values));

  runForward(output, BATCH_SIZE);

  auto eigen_inputs = dataToEigen(BATCH_SIZE, DIM1, DIM2, DIM3, DIM3 / 2,
                                  input_indices, input_values);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < BATCH_SIZE * DIM1 * DIM2 * OUTPUT_DIM; i++) {
    ASSERT_FLOAT_EQ(eigen_output.data()[i],
                    output->tensor()->activationsPtr()[i]);
  }

  auto output_grads = randomValues(BATCH_SIZE * DIM1 * DIM2 * OUTPUT_DIM);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_output_grads =
      dataToEigen(BATCH_SIZE, DIM1, DIM2, OUTPUT_DIM, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < BATCH_SIZE * DIM1 * DIM2 * DIM3 / 2; i++) {
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[input_indices.at(i)],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  const float* weight_grads = op->gradients().at(0)->data();
  for (uint32_t i = 0; i < op->inputDim() * op->dim(); i++) {
    ASSERT_FLOAT_EQ(eigen_weight_grads.data()[i], weight_grads[i]);
  }
}

}  // namespace thirdai::bolt::nn::tests