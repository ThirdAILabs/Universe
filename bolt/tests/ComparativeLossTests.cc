#include "gtest/gtest.h"
#include <bolt/src/nn/loss/ComparativeLoss.h>

namespace thirdai::bolt::nn::tests {

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

class LossTracker final : public loss::ComparativeLoss {
 public:
  explicit LossTracker(tensor::ActivationTensorPtr activations)
      : loss::ComparativeLoss(std::move(activations)) {}

  const auto& lossCalledWith() const { return _loss_called_with; }

  const auto& gradientCalledWith() const { return _gradient_called_with; }

 private:
  float singleLoss(float activation, float label) const override {
    const_cast<LossTracker*>(this)->_loss_called_with.emplace_back(activation,
                                                                   label);
    return 0.0;
  }

  float singleGradient(float activation, float label,
                       uint32_t batch_size) const override {
    (void)batch_size;
    const_cast<LossTracker*>(this)->_gradient_called_with.emplace_back(
        activation, label);
    return 0.0;
  }

  std::vector<std::pair<float, float>> _loss_called_with;
  std::vector<std::pair<float, float>> _gradient_called_with;
};

BoltVector sparseLabel() {
  return BoltVector::makeSparseVector({1, 4, 5, 6}, {1.0, 2.0, 0.0, 3.0});
}

BoltVector denseLabel() {
  return BoltVector::makeDenseVector({0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0});
}

tensor::ActivationTensorPtr sparseOutput() {
  auto output = tensor::ActivationTensor::make(
      /* dim= */ 8, /* sparse_nonzeros= */ 4, /* source= */ nullptr);
  output->allocate(/* batch_size= */ 1, /* use_sparsity= */ true);

  std::vector<uint32_t> indices = {0, 1, 4, 7};
  std::vector<float> values = {0.0, 4.0, 5.0, 6.0};

  uint32_t* active_neurons = output->getVector(0).active_neurons;
  std::copy(indices.begin(), indices.end(), active_neurons);

  float* activations = output->getVector(0).activations;
  std::copy(values.begin(), values.end(), activations);

  return output;
}

tensor::ActivationTensorPtr denseOutput() {
  auto output = tensor::ActivationTensor::make(
      /* dim= */ 8, /* sparse_nonzeros= */ 8, /* source= */ nullptr);
  output->allocate(/* batch_size= */ 1, /* use_sparsity= */ true);

  std::vector<float> values = {0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0};

  float* activations = output->getVector(0).activations;
  std::copy(values.begin(), values.end(), activations);

  return output;
}

void runTest(bool output_sparse, bool label_sparse, bool test_loss,
             const std::vector<std::pair<float, float>>& expected_called_with) {
  auto output = output_sparse ? sparseOutput() : denseOutput();

  BoltVector label = label_sparse ? sparseLabel() : denseLabel();

  BoltBatch label_batch({label});

  LossTracker loss(output);

  loss.labels()->setInputs(label_batch);

  std::vector<std::pair<float, float>> called_with;
  if (test_loss) {
    loss.loss(/* index_in_batch= */ 0);
    called_with = loss.lossCalledWith();
  } else {
    loss.gradients(/* index_in_batch= */ 0, /* batch_size= */ 1);
    called_with = loss.gradientCalledWith();
  }

  ASSERT_EQ(called_with.size(), expected_called_with.size());

  for (uint32_t i = 0; i < called_with.size(); i++) {
    ASSERT_EQ(called_with.at(i).first, expected_called_with.at(i).first);
    ASSERT_EQ(called_with.at(i).second, expected_called_with.at(i).second);
  }
}

TEST(ComparativeLossTest, LossDenseOutputDenseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};

  runTest(/* output_sparse= */ false, /* label_sparse= */ false,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTest, LossDenseOutputSparseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};

  runTest(/* output_sparse= */ false, /* label_sparse= */ true,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTest, LossSparseOutputDenseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ false,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTest, LossSparseOutputSparseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {5.0, 2.0}, {6.0, 0.0}, {0.0, 0.0}, {0.0, 3.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ true,
          /* test_loss= */ true, expected_called_with);
}

TEST(ComparativeLossTest, GradientDenseOutputDenseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ false, /* label_sparse= */ false,
          /* test_loss= */ false, expected_called_with);
}

TEST(ComparativeLossTest, GradientDenseOutputSparseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
      {5.0, 2.0}, {0.0, 0.0}, {0.0, 3.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ false, /* label_sparse= */ true,
          /* test_loss= */ false, expected_called_with);
}

TEST(ComparativeLossTest, GradientSparseOutputDenseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {5.0, 2.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ false,
          /* test_loss= */ false, expected_called_with);
}

TEST(ComparativeLossTest, GradientSparseOutputSparseLabels) {
  std::vector<std::pair<float, float>> expected_called_with = {
      {0.0, 0.0}, {4.0, 1.0}, {5.0, 2.0}, {6.0, 0.0}};
  runTest(/* output_sparse= */ true, /* label_sparse= */ true,
          /* test_loss= */ false, expected_called_with);
}

}  // namespace thirdai::bolt::nn::tests