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

void checkDenseOutput(const EigenMatrix& eigen_output,
                      const tensor::TensorPtr& bolt_output) {
  for (uint32_t i = 0; i < N_VECS * OUTPUT_DIM; i++) {
    ASSERT_FLOAT_EQ(eigen_output.data()[i], bolt_output->activationsPtr()[i]);
  }
}

void checkWeightGrads(const EigenMatrix& eigen_weight_grads,
                      const ops::FullyConnectedPtr& op) {
  const float* weight_grads = op->gradients().at(0)->data();
  for (uint32_t i = 0; i < op->inputDim() * op->dim(); i++) {
    ASSERT_FLOAT_EQ(eigen_weight_grads.data()[i], weight_grads[i]);
  }
}

TEST(FullyConnectedOpTests, DenseDense) {
  auto [input, op, output] = makeOp(1.0);

  auto input_data = randomValues(N_VECS, INPUT_DIM);

  input->setTensor(dataToTensor(INPUT_TENSOR_DIMS, input_data));

  runForward(output, BATCH_SIZE);

  auto eigen_inputs = dataToEigen(N_VECS, INPUT_DIM, input_data);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  checkDenseOutput(eigen_output, output->tensor());

  auto output_grads = randomValues(N_VECS, OUTPUT_DIM);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_output_grads = dataToEigen(N_VECS, OUTPUT_DIM, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < N_VECS * INPUT_DIM; i++) {
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[i],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  checkWeightGrads(eigen_weight_grads, op);
}

TEST(FullyConnectedOpTests, SparseDense) {
  auto [input, op, output] = makeOp(1.0);

  auto input_indices = randomIndices(N_VECS, INPUT_DIM, INPUT_NONZEROS);
  auto input_values = randomValues(N_VECS, INPUT_NONZEROS);

  input->setTensor(dataToTensor(INPUT_TENSOR_DIMS, INPUT_NONZEROS,
                                input_indices, input_values));

  runForward(output, BATCH_SIZE);

  auto eigen_inputs = dataToEigen(N_VECS, INPUT_DIM, INPUT_NONZEROS,
                                  input_indices, input_values);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  checkDenseOutput(eigen_output, output->tensor());

  auto output_grads = randomValues(N_VECS, OUTPUT_DIM);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_output_grads = dataToEigen(N_VECS, OUTPUT_DIM, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < N_VECS * INPUT_NONZEROS; i++) {
    uint32_t index = (i / INPUT_NONZEROS * INPUT_DIM) + input_indices.at(i);
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[index],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  checkWeightGrads(eigen_weight_grads, op);
}

TEST(FullyConnectedOpTests, DenseSparse) {
  auto [input, op, output] = makeOp(0.5);

  auto labels = ops::Input::make({INNER_DIM_1, INNER_DIM_2, OUTPUT_DIM});

  output->addInput(labels);

  auto input_data = randomValues(N_VECS, INPUT_DIM);

  input->setTensor(dataToTensor(INPUT_TENSOR_DIMS, input_data));

  auto output_indices = randomIndices(N_VECS, OUTPUT_DIM, OUTPUT_NONZEROS);
  auto output_label_vals = std::vector<float>(N_VECS * OUTPUT_NONZEROS, 1.0);

  labels->setTensor(dataToTensor(OUTPUT_TENSOR_DIMS, OUTPUT_NONZEROS,
                                 output_indices, output_label_vals));

  runForward(output, BATCH_SIZE);

  auto eigen_inputs = dataToEigen(N_VECS, INPUT_DIM, input_data);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < N_VECS; i++) {
    for (uint32_t j = 0; j < OUTPUT_NONZEROS; j++) {
      ASSERT_FLOAT_EQ(
          eigen_output.data()[i * OUTPUT_DIM +
                              output_indices.at(i * OUTPUT_NONZEROS + j)],
          output->tensor()->activationsPtr()[i * OUTPUT_NONZEROS + j]);
    }
  }

  auto output_grads = randomValues(N_VECS, OUTPUT_NONZEROS);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_output_grads = dataToEigen(N_VECS, OUTPUT_DIM, OUTPUT_NONZEROS,
                                        output_indices, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < N_VECS * INPUT_DIM; i++) {
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[i],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  checkWeightGrads(eigen_weight_grads, op);
}

TEST(FullyConnectedOpTests, SparseSparse) {
  auto [input, op, output] = makeOp(0.5);

  auto labels = ops::Input::make({INNER_DIM_1, INNER_DIM_2, OUTPUT_DIM});

  output->addInput(labels);

  auto input_indices = randomIndices(N_VECS, INPUT_DIM, INPUT_NONZEROS);
  auto input_values = randomValues(N_VECS, INPUT_NONZEROS);

  input->setTensor(dataToTensor(INPUT_TENSOR_DIMS, INPUT_NONZEROS,
                                input_indices, input_values));

  auto output_indices = randomIndices(N_VECS, OUTPUT_DIM, OUTPUT_NONZEROS);
  auto output_label_vals = std::vector<float>(N_VECS * OUTPUT_NONZEROS, 1.0);

  labels->setTensor(dataToTensor(OUTPUT_TENSOR_DIMS, OUTPUT_NONZEROS,
                                 output_indices, output_label_vals));

  runForward(output, BATCH_SIZE);

  auto eigen_inputs = dataToEigen(N_VECS, INPUT_DIM, INPUT_NONZEROS,
                                  input_indices, input_values);

  auto eigen_weights = weightsToEigen(op);

  auto eigen_output =
      eigenForward(eigen_inputs, eigen_weights, biasesToEigen(op));

  for (uint32_t i = 0; i < N_VECS; i++) {
    for (uint32_t j = 0; j < OUTPUT_NONZEROS; j++) {
      ASSERT_FLOAT_EQ(
          eigen_output.data()[i * OUTPUT_DIM +
                              output_indices.at(i * OUTPUT_NONZEROS + j)],
          output->tensor()->activationsPtr()[i * OUTPUT_NONZEROS + j]);
    }
  }

  auto output_grads = randomValues(N_VECS, OUTPUT_NONZEROS);
  setGradients(output->tensor(), output_grads);

  runBackpropagate(output, BATCH_SIZE);

  auto eigen_output_grads = dataToEigen(N_VECS, OUTPUT_DIM, OUTPUT_NONZEROS,
                                        output_indices, output_grads);

  auto eigen_input_grads =
      eigenBackpropagateInputGrads(eigen_output_grads, eigen_weights);

  for (uint32_t i = 0; i < N_VECS * INPUT_NONZEROS; i++) {
    uint32_t index = (i / INPUT_NONZEROS * INPUT_DIM) + input_indices.at(i);
    ASSERT_FLOAT_EQ(eigen_input_grads.data()[index],
                    input->tensor()->gradientsPtr()[i]);
  }

  auto eigen_weight_grads =
      eigenBackpropagateWeightGrads(eigen_inputs, eigen_output_grads);

  checkWeightGrads(eigen_weight_grads, op);
}

}  // namespace thirdai::bolt::nn::tests