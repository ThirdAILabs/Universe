#include "TestUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace thirdai::bolt::tests {

// Helper function. Fills the active neurons and activations for each vector
// with the index of the vector in the tensor and the gradients with 2 * the
// index of the vector in the tensor.
void fillTensor(TensorPtr& tensor) {
  for (uint32_t i = 0; i < tensor->batchSize(); i++) {
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
void checkTensorContents(const TensorPtr& tensor) {
  uint32_t nonzeros = tensor->nonzeros().value();
  for (uint32_t i = 0; i < tensor->batchSize(); i++) {
    for (uint32_t j = 0; j < nonzeros; j++) {
      if (nonzeros < tensor->dim()) {
        uint32_t active_neuron = tensor->activeNeuronsPtr()[i * nonzeros + j];
        ASSERT_EQ(active_neuron, i);
      }

      float activation = tensor->activationsPtr()[i * nonzeros + j];
      ASSERT_EQ(activation, static_cast<float>(i));

      float gradient = tensor->gradientsPtr()[i * nonzeros + j];
      ASSERT_EQ(gradient, 2 * static_cast<float>(i));
    }
  }
}

TEST(TensorTests, DenseTensor) {
  auto tensor = Tensor::dense(/* batch_size= */ 4, /* dim= */ 10);

  EXPECT_EQ(tensor->batchSize(), 4);
  EXPECT_EQ(tensor->dim(), 10);
  EXPECT_TRUE(tensor->nonzeros().has_value());
  EXPECT_EQ(tensor->nonzeros().value(), 10);

  EXPECT_EQ(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_NE(tensor->activationsPtr(), nullptr);
  EXPECT_NE(tensor->gradientsPtr(), nullptr);

  fillTensor(tensor);
  checkTensorContents(tensor);
}

TEST(TensorTests, SparseTensor) {
  auto tensor = Tensor::sparse(/* batch_size= */ 4, /* dim= */ 10,
                               /* nonzeros= */ 5);

  EXPECT_EQ(tensor->batchSize(), 4);
  EXPECT_EQ(tensor->dim(), 10);
  EXPECT_TRUE(tensor->nonzeros().has_value());
  EXPECT_EQ(tensor->nonzeros().value(), 5);

  EXPECT_NE(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_NE(tensor->activationsPtr(), nullptr);
  EXPECT_NE(tensor->gradientsPtr(), nullptr);

  fillTensor(tensor);
  checkTensorContents(tensor);
}

TEST(TensorTests, SparseTensorFromIndicesValues) {
  std::vector<uint32_t> indices(12);
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<float> values(12);
  std::iota(values.begin(), values.end(), 0);

  std::vector<size_t> lens = {5, 3, 4};
  auto lens_copy = lens;
  auto tensor = Tensor::sparse(std::move(indices), std::move(values),
                               std::move(lens_copy), /* dim= */ 12);

  EXPECT_EQ(tensor->batchSize(), 3);
  EXPECT_EQ(tensor->dim(), 12);
  EXPECT_FALSE(tensor->nonzeros().has_value());

  EXPECT_NE(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_NE(tensor->activationsPtr(), nullptr);
  EXPECT_EQ(tensor->gradientsPtr(), nullptr);

  size_t cnt = 0;
  for (uint32_t vec_idx = 0; vec_idx < 3; vec_idx++) {
    const BoltVector& vec = tensor->getVector(vec_idx);
    EXPECT_EQ(vec.len, lens[vec_idx]);

    EXPECT_FALSE(vec.isDense());
    EXPECT_FALSE(vec.hasGradients());
    EXPECT_EQ(vec.gradients, nullptr);

    for (size_t i = 0; i < lens[vec_idx]; i++) {
      EXPECT_EQ(vec.active_neurons[i], cnt);
      EXPECT_EQ(vec.activations[i], static_cast<float>(cnt));
      cnt++;
    }
  }
}

TEST(TensorTests, TensorFromArray) {
  std::vector<uint32_t> indices = {2, 1, 8, 7, 4, 6, 0, 9, 3, 5, 1, 4};
  std::vector<float> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto tensor = Tensor::fromArray(indices.data(), values.data(), 4, 10, 3,
                                  /* with_grad= */ false);

  EXPECT_EQ(tensor->batchSize(), 4);
  EXPECT_EQ(tensor->dim(), 10);
  EXPECT_TRUE(tensor->nonzeros().has_value());
  EXPECT_EQ(tensor->nonzeros().value(), 3);

  std::vector<BoltVector> expected_vecs = {
      BoltVector::makeSparseVector({2, 1, 8}, {1, 2, 3}),
      BoltVector::makeSparseVector({7, 4, 6}, {4, 5, 6}),
      BoltVector::makeSparseVector({0, 9, 3}, {7, 8, 9}),
      BoltVector::makeSparseVector({5, 1, 4}, {10, 11, 12}),
  };

  for (size_t i = 0; i < 4; i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        tensor->getVector(i), expected_vecs[i]);
  }
}

TEST(TensorTests, ConvertDenseBoltBatchToTensor) {
  std::vector<BoltVector> vectors = {
      BoltVector::makeDenseVector({1.0, 2.0, 3.0, 4.0}),
      BoltVector::makeDenseVector({5.0, 6.0, 7.0, 8.0}),
      BoltVector::makeDenseVector({9.0, 10.0, 11.0, 12.0})};

  auto vectors_copy = vectors;
  BoltBatch batch(std::move(vectors_copy));

  auto tensor = Tensor::convert(std::move(batch), 4);

  EXPECT_EQ(tensor->batchSize(), 3);
  EXPECT_EQ(tensor->dim(), 4);
  EXPECT_FALSE(tensor->nonzeros().has_value());

  EXPECT_EQ(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_EQ(tensor->activationsPtr(), nullptr);
  EXPECT_EQ(tensor->gradientsPtr(), nullptr);

  for (uint32_t i = 0; i < vectors.size(); i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        tensor->getVector(i), vectors[i]);
  }
}

TEST(TensorTests, CopyDenseBoltBatchToTensor) {
  std::vector<BoltVector> vectors = {
      BoltVector::makeDenseVector({1.0, 2.0, 3.0, 4.0}),
      BoltVector::makeDenseVector({5.0, 6.0, 7.0, 8.0}),
      BoltVector::makeDenseVector({9.0, 10.0, 11.0, 12.0})};

  auto vectors_copy = vectors;
  BoltBatch batch(std::move(vectors_copy));

  auto tensor = Tensor::copy(batch, 4);

  EXPECT_EQ(tensor->batchSize(), 3);
  EXPECT_EQ(tensor->dim(), 4);
  EXPECT_FALSE(tensor->nonzeros().has_value());

  EXPECT_EQ(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_NE(tensor->activationsPtr(), nullptr);
  EXPECT_EQ(tensor->gradientsPtr(), nullptr);

  for (uint32_t i = 0; i < vectors.size(); i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        tensor->getVector(i), vectors[i]);
  }

  std::vector<float> expected_values = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                        7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

  for (uint32_t i = 0; i < expected_values.size(); i++) {
    EXPECT_EQ(tensor->activationsPtr()[i], expected_values.at(i));
  }
}

TEST(TensorTests, ConvertSparseBoltBatchToTensor) {
  std::vector<BoltVector> vectors = {
      BoltVector::makeSparseVector({3, 0, 5, 7}, {1.0, 2.0, 3.0, 4.0}),
      BoltVector::makeSparseVector({1, 3, 6, 4, 4}, {5.0, 6.0, 7.0, 8.0, 9.0}),
      BoltVector::makeSparseVector({5, 2, 0}, {10.0, 11.0, 12.0})};

  auto vectors_copy = vectors;
  BoltBatch batch(std::move(vectors_copy));

  auto tensor = Tensor::convert(std::move(batch), 8);

  EXPECT_EQ(tensor->batchSize(), 3);
  EXPECT_EQ(tensor->dim(), 8);
  EXPECT_FALSE(tensor->nonzeros().has_value());

  EXPECT_EQ(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_EQ(tensor->activationsPtr(), nullptr);
  EXPECT_EQ(tensor->gradientsPtr(), nullptr);

  for (uint32_t i = 0; i < vectors.size(); i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        tensor->getVector(i), vectors[i]);
  }
}

TEST(TensorTests, CopySparseBoltBatchToTensor) {
  std::vector<BoltVector> vectors = {
      BoltVector::makeSparseVector({3, 0, 5, 7}, {1.0, 2.0, 3.0, 4.0}),
      BoltVector::makeSparseVector({1, 3, 6, 4, 4}, {5.0, 6.0, 7.0, 8.0, 9.0}),
      BoltVector::makeSparseVector({5, 2, 0}, {10.0, 11.0, 12.0})};

  auto vectors_copy = vectors;
  BoltBatch batch(std::move(vectors_copy));

  auto tensor = Tensor::copy(batch, 8);

  EXPECT_EQ(tensor->batchSize(), 3);
  EXPECT_EQ(tensor->dim(), 8);
  EXPECT_FALSE(tensor->nonzeros().has_value());

  EXPECT_NE(tensor->activeNeuronsPtr(), nullptr);
  EXPECT_NE(tensor->activationsPtr(), nullptr);
  EXPECT_EQ(tensor->gradientsPtr(), nullptr);

  for (uint32_t i = 0; i < vectors.size(); i++) {
    thirdai::tests::BoltVectorTestUtils::assertBoltVectorsAreEqual(
        tensor->getVector(i), vectors[i]);
  }

  std::vector<uint32_t> expected_indices = {3, 0, 5, 7, 1, 3, 6, 4, 4, 5, 2, 0};
  std::vector<float> expected_values = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                        7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

  for (uint32_t i = 0; i < expected_indices.size(); i++) {
    EXPECT_EQ(tensor->activeNeuronsPtr()[i], expected_indices.at(i));
  }

  for (uint32_t i = 0; i < expected_values.size(); i++) {
    EXPECT_EQ(tensor->activationsPtr()[i], expected_values.at(i));
  }
}

TEST(TensorTests, MismatchedSparseDenseVectorsError) {
  BoltBatch batch({BoltVector::makeSparseVector({1}, {1.0}),
                   BoltVector::makeDenseVector({1.0, 2.0})});
  // NOLINTNEXTLINE
  ASSERT_THROW(Tensor::convert(std::move(batch), 2), std::invalid_argument);
}

TEST(TensorTests, DenseVectorDimMismatch) {
  BoltBatch batch({BoltVector::makeDenseVector({1.0, 2.0})});

  // NOLINTNEXTLINE
  ASSERT_THROW(Tensor::convert(std::move(batch), 3), std::invalid_argument);
}

TEST(TensorTests, SparseVectorDimMismatch) {
  BoltBatch batch({BoltVector::makeSparseVector({1, 4}, {1.0, 1.0})});

  // NOLINTNEXTLINE
  ASSERT_THROW(Tensor::convert(std::move(batch), 2), std::invalid_argument);
}

}  // namespace thirdai::bolt::tests