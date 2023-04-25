#include <wrappers/src/EigenDenseWrapper.h>
#include "gtest/gtest.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <Eigen/src/Core/Matrix.h>
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

  op->setWeightsAndBiases(weights.data(), biases.data());
}

void setGradients(tensor::TensorPtr& tensor, const std::vector<float>& grads) {
  float* grad_ptr = const_cast<float*>(tensor->gradientsPtr());

  std::copy(grads.begin(), grads.end(), grad_ptr);
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

uint32_t dim1 = 2, dim2 = 1, dim3 = 1, dim4 = 10;
// uint32_t dim1 = 4, dim2 = 6, dim3 = 8, dim4 = 10;

uint32_t n_vecs = dim1 * dim2 * dim3;

std::vector<uint32_t> dims = {dim1, dim2, dim3, dim4};
std::vector<uint32_t> input_dims = {dim2, dim3, dim4};

uint32_t batch_size = dim1, input_dim = dim4, output_dim = 20,
         input_nonzeros = dim4 / 2, output_nonzeros = output_dim / 2;

TEST(FullyConnectedOpTests, DenseDense) {
  auto input = ops::Input::make(input_dims);

  auto op = ops::FullyConnected::make(/* dim= */ output_dim,
                                      /* input_dim= */ input_dim,
                                      /* sparsity= */ 1.0,
                                      /* activation= */ "linear");
  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->allocate(batch_size, /* use_sparsity= */ true);

  auto input_data = randomValues(n_vecs, input_dim);

  input->setTensor(dataToTensor(dims, input_data));

  runForward(output, batch_size);

  auto eigen_inputs = dataToEigen(n_vecs, input_dim, input_data);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < n_vecs * output_dim; i++) {
    ASSERT_FLOAT_EQ(eigen_output.data()[i],
                    output->tensor()->activationsPtr()[i]);
  }

  auto output_grads = randomValues(n_vecs, output_dim);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, batch_size);

  auto eigen_output_grads = dataToEigen(n_vecs, output_dim, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < n_vecs * input_dim; i++) {
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
  auto input = ops::Input::make(input_dims);

  auto op = ops::FullyConnected::make(/* dim= */ output_dim,
                                      /* input_dim= */ input_dim,
                                      /* sparsity= */ 1.0,
                                      /* activation= */ "linear");
  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->allocate(batch_size, /* use_sparsity= */ true);

  auto input_indices = randomIndices(n_vecs, input_dim, input_nonzeros);
  auto input_values = randomValues(n_vecs, input_nonzeros);

  input->setTensor(
      dataToTensor(dims, input_nonzeros, input_indices, input_values));

  runForward(output, batch_size);

  auto eigen_inputs = dataToEigen(n_vecs, input_dim, input_nonzeros,
                                  input_indices, input_values);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < n_vecs * output_dim; i++) {
    ASSERT_FLOAT_EQ(eigen_output.data()[i],
                    output->tensor()->activationsPtr()[i]);
  }

  auto output_grads = randomValues(n_vecs, output_dim);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, batch_size);

  auto eigen_output_grads = dataToEigen(n_vecs, output_dim, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < n_vecs * input_nonzeros; i++) {
    uint32_t index = (i / input_nonzeros * input_dim) + input_indices.at(i);
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[index],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  const float* weight_grads = op->gradients().at(0)->data();
  for (uint32_t i = 0; i < op->inputDim() * op->dim(); i++) {
    ASSERT_FLOAT_EQ(eigen_weight_grads.data()[i], weight_grads[i]);
  }
}

TEST(FullyConnectedOpTests, DenseSparse) {
  auto input = ops::Input::make(input_dims);
  auto labels = ops::Input::make({dim2, dim3, output_dim});

  auto op = ops::FullyConnected::make(/* dim= */ output_dim,
                                      /* input_dim= */ input_dim,
                                      /* sparsity= */ 0.5,
                                      /* activation= */ "linear");

  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->addInput(labels);

  output->allocate(batch_size, /* use_sparsity= */ true);

  auto input_data = randomValues(n_vecs, input_dim);

  input->setTensor(dataToTensor(dims, input_data));

  auto output_indices = randomIndices(n_vecs, output_dim, output_nonzeros);
  auto output_label_vals = std::vector<float>(n_vecs * output_nonzeros, 1.0);

  labels->setTensor(dataToTensor({dim1, dim2, dim3, output_dim},
                                 output_nonzeros, output_indices,
                                 output_label_vals));

  runForward(output, batch_size);

  auto eigen_inputs = dataToEigen(n_vecs, input_dim, input_data);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < n_vecs; i++) {
    for (uint32_t j = 0; j < output_nonzeros; j++) {
      ASSERT_FLOAT_EQ(
          eigen_output.data()[i * output_dim +
                              output_indices.at(i * output_nonzeros + j)],
          output->tensor()->activationsPtr()[i * output_nonzeros + j]);
    }
  }

  auto output_grads = randomValues(n_vecs, output_nonzeros);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, batch_size);

  auto eigen_output_grads = dataToEigen(n_vecs, output_dim, output_nonzeros,
                                        output_indices, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < n_vecs * input_dim; i++) {
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

TEST(FullyConnectedOpTests, SparseSparse) {
  auto input = ops::Input::make(input_dims);
  auto labels = ops::Input::make({dim2, dim3, output_dim});

  auto op = ops::FullyConnected::make(/* dim= */ output_dim,
                                      /* input_dim= */ input_dim,
                                      /* sparsity= */ 0.5,
                                      /* activation= */ "linear");

  setWeightsAndBiases(op);

  auto output = op->apply(input);
  output->addInput(labels);

  output->allocate(batch_size, /* use_sparsity= */ true);

  auto input_indices = randomIndices(n_vecs, input_dim, input_nonzeros);
  auto input_values = randomValues(n_vecs, input_nonzeros);

  input->setTensor(
      dataToTensor(dims, input_nonzeros, input_indices, input_values));

  auto output_indices = randomIndices(n_vecs, output_dim, output_nonzeros);
  auto output_label_vals = std::vector<float>(n_vecs * output_nonzeros, 1.0);

  labels->setTensor(dataToTensor({dim1, dim2, dim3, output_dim},
                                 output_nonzeros, output_indices,
                                 output_label_vals));

  runForward(output, batch_size);

  auto eigen_inputs = dataToEigen(n_vecs, input_dim, input_nonzeros,
                                  input_indices, input_values);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < n_vecs; i++) {
    for (uint32_t j = 0; j < output_nonzeros; j++) {
      ASSERT_FLOAT_EQ(
          eigen_output.data()[i * output_dim +
                              output_indices.at(i * output_nonzeros + j)],
          output->tensor()->activationsPtr()[i * output_nonzeros + j]);
    }
  }

  auto output_grads = randomValues(n_vecs, output_nonzeros);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, batch_size);

  auto eigen_output_grads = dataToEigen(n_vecs, output_dim, output_nonzeros,
                                        output_indices, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < n_vecs * input_nonzeros; i++) {
    uint32_t index = (i / input_nonzeros * input_dim) + input_indices.at(i);
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[index],
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