#include "gtest/gtest.h"
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>

namespace thirdai::bolt::nn::tests {

BoltBatch sparseInputBatch() {
  return BoltBatch({
      BoltVector::makeSparseVectorWithGradients({1, 3, 4, 6},
                                                {1.0, 3.0, 4.0, 6.0}),
      BoltVector::makeSparseVectorWithGradients({0, 2, 5, 7},
                                                {0.0, 2.0, 5.0, 7.0}),
  });
}

BoltBatch denseInputBatch() {
  return BoltBatch({
      BoltVector::makeDenseVectorWithGradients({1.0, 2.0, 3.0, 4.0, 5.0}),
      BoltVector::makeDenseVectorWithGradients({6.0, 7.0, 8.0, 9.0, 10.0}),
  });
}

tensor::InputTensorPtr getInputTensor(bool dense) {
  if (dense) {
    return tensor::InputTensor::make(/* dim= */ 5, /* num_nonzeros= */ 5);
  }
  return tensor::InputTensor::make(/* dim= */ 8, /* num_nonzeros= */ 4);
}

std::tuple<std::vector<BoltVector>, BoltBatch, BoltBatch>
runConcatenateForwardBackpropagate(
    bool input_1_dense, bool input_2_dense,
    const std::vector<std::vector<float>>& output_gradients) {
  auto input_1 = getInputTensor(input_1_dense);
  auto input_2 = getInputTensor(input_2_dense);

  auto output = ops::Concatenate::make()->apply({input_1, input_2});
  output->allocate(/* batch_size= */ 2, /* use_sparsity= */ true);

  BoltBatch batch_1 = input_1_dense ? denseInputBatch() : sparseInputBatch();
  BoltBatch batch_2 = input_2_dense ? denseInputBatch() : sparseInputBatch();

  input_1->setInputs(batch_1);
  input_2->setInputs(batch_2);

  std::vector<BoltVector> output_vectors;
  for (uint32_t i = 0; i < 2; i++) {
    output->forward(/* index_in_batch= */ i, /* training= */ true);
    output_vectors.push_back(output->getVector(i));

    EXPECT_EQ(output_gradients[i].size(), output->getVector(i).len);
    std::copy(output_gradients[i].begin(), output_gradients[i].end(),
              output->getVector(i).gradients);

    output->backpropagate(/* index_in_batch= */ i);
  }

  return {output_vectors, std::move(batch_1), std::move(batch_2)};
}

void verifyInputsAndOutputs(
    const std::vector<BoltVector>& outputs,
    const std::vector<BoltVector>& expected_outputs, const BoltBatch& input_1,
    bool input_1_dense, const BoltBatch& input_2, bool input_2_dense,
    const std::vector<std::vector<float>>& output_gradients) {
  BoltBatch expected_input_1 =
      input_1_dense ? denseInputBatch() : sparseInputBatch();
  BoltBatch expected_input_2 =
      input_2_dense ? denseInputBatch() : sparseInputBatch();

  uint32_t gradient_split_index = input_1_dense ? 5 : 4;
  for (uint32_t i = 0; i < 2; i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        outputs[i], expected_outputs[i]);

    EXPECT_EQ(gradient_split_index, expected_input_1[i].len);
    std::copy(output_gradients[i].begin(),
              output_gradients[i].begin() + gradient_split_index,
              expected_input_1[i].gradients);

    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        input_1[i], expected_input_1[i]);

    EXPECT_EQ(output_gradients[i].size() - gradient_split_index,
              expected_input_2[i].len);
    std::copy(output_gradients[i].begin() + gradient_split_index,
              output_gradients[i].end(), expected_input_2[i].gradients);

    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        input_2[i], expected_input_2[i]);
  }
}

TEST(ConcatenateOpTests, DenseDense) {
  std::vector<std::vector<float>> output_gradients = {
      {6.0, 3.0, -1.0, 40.0, 23.0, 9.0, -37.0, 55.0, 9.0, -8.0},
      {50.0, 26.0, -3.0, 11.0, -54.0, 47.0, 7.0, -18.0, 76.0, 89.0},
  };

  auto [outputs, input_1, input_2] = runConcatenateForwardBackpropagate(
      /* input_1_dense= */ true, /* input_2_dense= */ true, output_gradients);

  std::vector<BoltVector> expected_outputs = {
      BoltVector::makeDenseVectorWithGradients(
          {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0}),
      BoltVector::makeDenseVectorWithGradients(
          {6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
  };

  verifyInputsAndOutputs(
      /* outputs= */ outputs, /* expected_outputs= */ expected_outputs,
      /* input_1= */ input_1, /* input_1_dense= */ true,
      /* input_2= */ input_2, /* input_2_dense= */ true,
      /* output_gradients= */ output_gradients);
}

TEST(ConcatenateOpTests, SparseDense) {
  std::vector<std::vector<float>> output_gradients = {
      {6.0, 3.0, -1.0, 23.0, 9.0, -37.0, 55.0, 9.0, -8.0},
      {50.0, 26.0, -3.0, 11.0, -54.0, 7.0, -18.0, 76.0, 89.0},
  };

  auto [outputs, input_1, input_2] = runConcatenateForwardBackpropagate(
      /* input_1_dense= */ false, /* input_2_dense= */ true, output_gradients);

  std::vector<BoltVector> expected_outputs = {
      BoltVector::makeSparseVectorWithGradients(
          {1, 3, 4, 6, 8, 9, 10, 11, 12},
          {1.0, 3.0, 4.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0}),
      BoltVector::makeSparseVectorWithGradients(
          {0, 2, 5, 7, 8, 9, 10, 11, 12},
          {0.0, 2.0, 5.0, 7.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
  };

  verifyInputsAndOutputs(
      /* outputs= */ outputs, /* expected_outputs= */ expected_outputs,
      /* input_1= */ input_1, /* input_1_dense= */ false,
      /* input_2= */ input_2, /* input_2_dense= */ true,
      /* output_gradients= */ output_gradients);
}

TEST(ConcatenateOpTests, DenseSparse) {
  std::vector<std::vector<float>> output_gradients = {
      {6.0, 3.0, -1.0, 23.0, 9.0, -37.0, 55.0, 9.0, -8.0},
      {50.0, 26.0, -3.0, 11.0, -54.0, 7.0, -18.0, 76.0, 89.0},
  };

  auto [outputs, input_1, input_2] = runConcatenateForwardBackpropagate(
      /* input_1_dense= */ true, /* input_2_dense= */ false, output_gradients);

  std::vector<BoltVector> expected_outputs = {
      BoltVector::makeSparseVectorWithGradients(
          {0, 1, 2, 3, 4, 6, 8, 9, 11},
          {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 3.0, 4.0, 6.0}),
      BoltVector::makeSparseVectorWithGradients(
          {0, 1, 2, 3, 4, 5, 7, 10, 12},
          {6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 2.0, 5.0, 7.0}),
  };

  verifyInputsAndOutputs(
      /* outputs= */ outputs, /* expected_outputs= */ expected_outputs,
      /* input_1= */ input_1, /* input_1_dense= */ true,
      /* input_2= */ input_2, /* input_2_dense= */ false,
      /* output_gradients= */ output_gradients);
}

TEST(ConcatenateOpTests, SparseSparse) {
  std::vector<std::vector<float>> output_gradients = {
      {6.0, -1.0, 23.0, 9.0, -37.0, 55.0, 9.0, -8.0},
      {26.0, -3.0, 11.0, -54.0, 7.0, -18.0, 76.0, 89.0},
  };

  auto [outputs, input_1, input_2] = runConcatenateForwardBackpropagate(
      /* input_1_dense= */ false, /* input_2_dense= */ false, output_gradients);

  std::vector<BoltVector> expected_outputs = {
      BoltVector::makeSparseVectorWithGradients(
          {1, 3, 4, 6, 9, 11, 12, 14},
          {1.0, 3.0, 4.0, 6.0, 1.0, 3.0, 4.0, 6.0}),
      BoltVector::makeSparseVectorWithGradients(
          {0, 2, 5, 7, 8, 10, 13, 15},
          {0.0, 2.0, 5.0, 7.0, 0.0, 2.0, 5.0, 7.0}),
  };

  verifyInputsAndOutputs(
      /* outputs= */ outputs, /* expected_outputs= */ expected_outputs,
      /* input_1= */ input_1, /* input_1_dense= */ false,
      /* input_2= */ input_2, /* input_2_dense= */ false,
      /* output_gradients= */ output_gradients);
}

}  // namespace thirdai::bolt::nn::tests