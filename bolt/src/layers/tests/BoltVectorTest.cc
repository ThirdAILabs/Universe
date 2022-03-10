#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <functional>

namespace thirdai::bolt::tests {

constexpr uint32_t VEC_LEN = 100;

void checkVectorsAfterCopy(const BoltVector& a, const BoltVector& b) {
  ASSERT_EQ(a.len, b.len);
  ASSERT_EQ(a.active_neurons == nullptr, b.active_neurons == nullptr);
  ASSERT_EQ(a.gradients == nullptr, b.gradients == nullptr);
  for (uint32_t i = 0; i < b.len; i++) {
    if (a.active_neurons != nullptr) {
      ASSERT_EQ(a.active_neurons[i], b.active_neurons[i]);
    }
    ASSERT_EQ(a.activations[i], b.activations[i]);
    if (a.gradients != nullptr) {
      ASSERT_EQ(a.gradients[i], b.gradients[i]);
    }
  }
}

void testCopy(BoltVector& a) {
  BoltVector b(a);  // NOLINT clang tidy complains about copy here
  checkVectorsAfterCopy(a, b);
}

void testCopyAssign(BoltVector& a) {
  BoltVector b = a;  // NOLINT clang tidy complains about copy here
  checkVectorsAfterCopy(a, b);
}

void checkVectorsAfterMove(const BoltVector& a, const BoltVector& b, bool dense,
                           bool has_grad) {
  ASSERT_EQ(a.len, 0);
  ASSERT_EQ(a.active_neurons, nullptr);
  ASSERT_EQ(a.activations, nullptr);
  ASSERT_EQ(a.gradients, nullptr);

  ASSERT_EQ(b.len, VEC_LEN);
  ASSERT_EQ(b.active_neurons == nullptr, dense);
  ASSERT_EQ(b.gradients != nullptr, has_grad);

  for (uint32_t i = 0; i < b.len; i++) {
    if (!dense) {
      ASSERT_EQ(b.active_neurons[i], i);
    }
    ASSERT_EQ(b.activations[i], i);
    if (has_grad) {
      ASSERT_EQ(b.gradients[i], i);
    }
  }
}

void testMove(BoltVector& a) {
  bool dense = a.isDense();
  bool has_grad = a.gradients != nullptr;

  BoltVector b(std::move(a));
  checkVectorsAfterMove(a, b, dense, has_grad);
}

void testMoveAssign(BoltVector& a) {
  bool dense = a.isDense();
  bool has_grad = a.gradients != nullptr;
  BoltVector b = std::move(a);
  checkVectorsAfterMove(a, b, dense, has_grad);
}

BoltVector makeVectorForTest(bool dense, bool has_grad) {
  BoltVector vec(VEC_LEN, dense, has_grad);
  for (uint32_t i = 0; i < VEC_LEN; i++) {
    if (!dense) {
      vec.active_neurons[i] = i;
    }
    vec.activations[i] = i;
    if (has_grad) {
      vec.gradients[i] = i;
    }
  }
  return vec;
}

void runTest(const std::function<void(BoltVector&)>& test_func) {
  {
    // Test sparse with gradients
    BoltVector a = makeVectorForTest(false, true);
    test_func(a);
  }

  {
    // Test dense with gradients
    BoltVector a = makeVectorForTest(true, true);
    test_func(a);
  }

  {
    // Test sparse with no gradients
    BoltVector a = makeVectorForTest(false, false);
    test_func(a);
  }

  {
    // Test dense with no gradients
    BoltVector a = makeVectorForTest(true, false);
    test_func(a);
  }

  {
    // Test sparse with gradients without owning data
    BoltVector a = makeVectorForTest(false, true);
    BoltVector b(a.active_neurons, a.activations, a.gradients, a.len);
    test_func(b);
  }

  {
    // Test dense with gradients without owning data
    BoltVector a = makeVectorForTest(true, true);
    BoltVector b(a.active_neurons, a.activations, a.gradients, a.len);
    test_func(b);
  }

  {
    // Test sparse with no gradients without owning data
    BoltVector a = makeVectorForTest(false, false);
    BoltVector b(a.active_neurons, a.activations, a.gradients, a.len);
    test_func(b);
  }

  {
    // Test dense with no gradients without owning data
    BoltVector a = makeVectorForTest(true, false);
    BoltVector b(a.active_neurons, a.activations, a.gradients, a.len);
    test_func(b);
  }
}

TEST(BoltVectorTests, CopyConstructor) { runTest(testCopy); }

TEST(BoltVectorTests, CopyAssignment) { runTest(testCopyAssign); }

TEST(BoltVectorTests, MoveConstructor) { runTest(testMove); }

TEST(BoltVectorTests, MoveAssignment) { runTest(testMoveAssign); }

}  // namespace thirdai::bolt::tests
