#include "TestUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <optional>

namespace thirdai::bolt::nn::tests {

void assertShapeEq(const std::vector<uint32_t>& shape,
                   const std::vector<uint32_t>& expected_shape) {
  ASSERT_EQ(shape.size(), expected_shape.size());

  for (uint32_t i = 0; i < shape.size(); i++) {
    ASSERT_EQ(shape.at(i), expected_shape.at(i));
  }
}

// Helper function. Fills the active neurons and activations for each vector
// with the index of the vector in the tensor and the gradients with 2 * the
// index of the vector in the tensor.
void fillTensor(tensor::ActivationTensorPtr& tensor) {
  for (uint32_t i = 0; i < tensor->shape()[0]; i++) {
    auto& vec = tensor->getVector(i);

    if (!vec.isDense()) {
      std::fill_n(vec.active_neurons, vec.len, i);
    }

    std::fill_n(vec.activations, vec.len, static_cast<float>(i));

    std::fill_n(vec.gradients, vec.len, static_cast<float>(2 * i));
  }
}

// Helper function. Checks that the pointers returned by the tensor point to
// memory that matches the result of call fillTensor on the tensor.
void checkTensorContents(const tensor::ActivationTensorPtr& tensor) {
  for (uint32_t i = 0; i < tensor->shape()[0]; i++) {
    for (uint32_t j = 0; j < tensor->shape()[1]; j++) {
      if (tensor->shape()[1] < tensor->dim()) {
        uint32_t active_neuron =
            tensor->activeNeuronsPtr()[i * tensor->shape()[1] + j];
        ASSERT_EQ(active_neuron, i);
      }

      float activation = tensor->activationsPtr()[i * tensor->shape()[1] + j];
      ASSERT_EQ(activation, static_cast<float>(i));

      float gradient = tensor->gradientsPtr()[i * tensor->shape()[1] + j];
      ASSERT_EQ(gradient, 2 * static_cast<float>(i));
    }
  }
}

// This test checks that the ActivationTensor allocates state correctly as the
// batch size and sparsity change.
TEST(TensorTests, ActivationTensor) {
  auto op = Noop::make("noop", /* dim= */ 8, /* num_nonzeros= */ 4);
  auto tensor = op->apply({emptyInput()});
  // auto tensor = tensor::ActivationTensor::make(
  //     /* dim= */ 8, /* sparse_nonzeros= */ 4, /* source= */ nullptr);

  /**
   * Start with sparse tensor with batch size 3.
   */

  tensor->allocate(/* batch_size= */ 3, /* use_sparsity= */ true);

  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ true), 4);
  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ false), 8);
  assertShapeEq(tensor->shape(), {3, 4});

  fillTensor(tensor);
  checkTensorContents(tensor);

  /**
   * Increase the batch size.
   */

  tensor->allocate(/* batch_size= */ 5, /* use_sparsity= */ false);

  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ true), 4);
  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ false), 8);
  assertShapeEq(tensor->shape(), {5, 8});

  fillTensor(tensor);
  checkTensorContents(tensor);

  /**
   * Update the sparsity but don't change the batch size, nothing should change.
   */

  op->updateNumNonzeros(/* new_num_nonzeros= */ 6);

  tensor->allocate(/* batch_size= */ 5, /* use_sparsity= */ false);

  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ true), 6);
  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ false), 8);
  assertShapeEq(tensor->shape(), {5, 8});

  fillTensor(tensor);
  checkTensorContents(tensor);

  /**
   * Now use the new sparsity value.
   */

  tensor->allocate(/* batch_size= */ 5, /* use_sparsity= */ true);

  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ true), 6);
  ASSERT_EQ(tensor->numNonzeros(/* use_sparsity= */ false), 8);
  assertShapeEq(tensor->shape(), {5, 6});

  fillTensor(tensor);
  checkTensorContents(tensor);
}

TEST(TensorTests, InputTensor) {
  std::vector<BoltVector> vectors = {
      BoltVector::makeDenseVector({1.0, 2.0, 3.0, 4.0}),
      BoltVector::makeSparseVector({1, 2, 3}, {1.0, 2.0, 3.0})};

  BoltBatch batch(std::move(vectors));

  auto tensor = tensor::InputTensor::make(/* dim= */ 4);

  tensor->setInputs(batch);

  for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        batch[i], tensor->getVector(i));
  }
}

}  // namespace thirdai::bolt::nn::tests