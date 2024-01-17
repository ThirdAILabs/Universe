#include "TestUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/loss/ComparativeLoss.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <cstddef>
#include <set>

namespace thirdai::bolt::tests {

/**
 * These tests check the correctness of the ComparativeLoss class in iterating
 * over the elements of the output and label vectors. To do this they use the
 * LossTracker class which is an instance of ComparativeLoss which records the
 * activation and label values singleLoss and singleGradient are called with. It
 * is then invoked with all combinations of sparse/dense output/labels to check
 * that with different sparsity combinations the loss function still matches
 * neurons and labels correctly and doesn't miss any elements of the loss or
 * gradient.
 */

class LossTracker final : public ComparativeLoss {
 public:
  explicit LossTracker(ComputationPtr activations, ComputationPtr labels)
      : ComparativeLoss(std::move(activations), std::move(labels)) {}

  const auto& lossCalledWith() const { return _loss_called_with; }

  const auto& gradientCalledWith() const { return _gradient_called_with; }

  ar::ConstArchivePtr toArchive() const final { return nullptr; }

 private:
  float singleLoss(float activation, float label) const override {
    const_cast<LossTracker*>(this)->_loss_called_with.insert(
        {activation, label});
    return 0.0;
  }

  float singleGradient(float activation, float label,
                       uint32_t batch_size) const override {
    (void)batch_size;
    const_cast<LossTracker*>(this)->_gradient_called_with.insert(
        {activation, label});
    return 0.0;
  }

  std::set<std::pair<float, float>> _loss_called_with;
  std::set<std::pair<float, float>> _gradient_called_with;
};

BoltVector sparseLabel() {
  return BoltVector::makeSparseVector({1, 4, 5, 6}, {1.0, 2.0, 0.0, 3.0});
}

BoltVector denseLabel() {
  return BoltVector::makeDenseVector({0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0});
}

TensorPtr sparseOutput() {
  auto output = Tensor::sparse(/* batch_size= */ 1, /* dim= */ 8,
                               /* nonzeros= */ 4);

  std::vector<uint32_t> indices = {0, 1, 4, 7};
  std::vector<float> values = {0.0, 4.0, 5.0, 6.0};

  uint32_t* active_neurons = output->getVector(0).active_neurons;
  std::copy(indices.begin(), indices.end(), active_neurons);

  float* activations = output->getVector(0).activations;
  std::copy(values.begin(), values.end(), activations);

  return output;
}

TensorPtr denseOutput() {
  auto output = Tensor::dense(/* batch_size= */ 1, /* dim= */ 8);

  std::vector<float> values = {0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0};

  float* activations = output->getVector(0).activations;
  std::copy(values.begin(), values.end(), activations);

  return output;
}

void runTest(bool output_sparse, bool label_sparse, bool test_loss,
             const std::set<std::pair<float, float>>& expected_called_with) {
  auto output_tensor = output_sparse ? sparseOutput() : denseOutput();

  auto output = Input::make(/* dim= */ 8);
  output->setTensor(output_tensor);

  BoltVector label_vector = label_sparse ? sparseLabel() : denseLabel();
  auto label_tensor = Tensor::convert(BoltBatch({label_vector}), /* dim= */ 8);

  auto label = Input::make(/* dim= */ 8);

  LossTracker loss(output, label);

  label->setTensor(label_tensor);

  std::set<std::pair<float, float>> called_with;
  if (test_loss) {
    loss.loss(/* index_in_batch= */ 0);
    called_with = loss.lossCalledWith();
  } else {
    loss.gradients(/* index_in_batch= */ 0, /* batch_size= */ 1);
    called_with = loss.gradientCalledWith();
  }

  ASSERT_EQ(called_with.size(), expected_called_with.size());

  for (const auto& expected : expected_called_with) {
    called_with.erase(expected);
  }
  ASSERT_EQ(called_with.size(), 0);
}

TEST(ComparativeLossTests, LossDenseOutputDenseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};

  runTest(/* output_sparse= */ false, /* label_sparse= */ false,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTests, LossDenseOutputSparseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};

  runTest(/* output_sparse= */ false, /* label_sparse= */ true,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTests, LossSparseOutputDenseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ false,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTests, LossSparseOutputSparseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {5.0, 2.0}, {6.0, 0.0}, {0.0, 0.0}, {0.0, 3.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ true,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTests, GradientDenseOutputDenseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ false, /* label_sparse= */ false,
          /* test_loss= */ false, expected_called_with);
}

TEST(ComparativeLossTests, GradientDenseOutputSparseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ false, /* label_sparse= */ true,
          /* test_loss= */ false, expected_called_with);
}

TEST(ComparativeLossTests, GradientSparseOutputDenseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {5.0, 2.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ false,
          /* test_loss= */ false, expected_called_with);
}

TEST(ComparativeLossTests, GradientSparseOutputSparseLabels) {
  std::set<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {5.0, 2.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ true,
          /* test_loss= */ false, expected_called_with);
}

}  // namespace thirdai::bolt::tests