#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/tests/BoltVectorTestUtils.h>
#include <gtest/gtest.h>
#include <functional>

namespace thirdai::tests {

constexpr uint32_t VEC_LEN = 100;

// a is the original vector, b is where it is copied.
void checkVectorEqualityAfterCopy(const BoltVector& a, const BoltVector& b) {
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
  checkVectorEqualityAfterCopy(a, b);
}

void testCopyAssign(BoltVector& a) {
  BoltVector b = a;  // NOLINT clang tidy complains about copy here
  checkVectorEqualityAfterCopy(a, b);
}

// a is the original vector, b is where it is moved.
void checkVectorEqualityAfterMove(const BoltVector& a, const BoltVector& b,
                                  bool dense, bool has_grad) {
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
  // Prevent linting because clang tidy doesn't like using a after move
  checkVectorEqualityAfterMove(a, b, dense, has_grad);  // NOLINT
}

void testMoveAssign(BoltVector& a) {
  bool dense = a.isDense();
  bool has_grad = a.gradients != nullptr;
  BoltVector b = std::move(a);
  // Prevent linting because clang tidy doesn't like using a after move
  checkVectorEqualityAfterMove(a, b, dense, has_grad);  // NOLINT
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
  // Note that the separate curly braces just make sure each vector is scoped
  // and so each check is independent without adding a bunch of additional unit
  // tests.
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

TEST(BoltVectorTests, DenseChunks) {
  BoltVector vec =
      BoltVector::makeDenseVectorWithGradients({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  auto chunk = vec.viewChunk(1, 2);
  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      chunk, BoltVector::makeDenseVectorWithGradients({3.0, 4.0}));

  chunk.activations[0] = 200;
  chunk.gradients[1] = -10;

  ASSERT_EQ(vec.activations[2], 200);
  ASSERT_EQ(vec.gradients[3], -10);
}

TEST(BoltVectorTests, SparseChunks) {
  BoltVector vec = BoltVector::makeSparseVectorWithGradients(
      {10, 20, 30, 40, 50, 60}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  auto chunk = vec.viewChunk(1, 2);
  BoltVectorTestUtils::assertBoltVectorsAreEqual(
      chunk, BoltVector::makeSparseVectorWithGradients({30, 40}, {3.0, 4.0}));

  chunk.active_neurons[1] = 1000;
  chunk.activations[0] = 200;
  chunk.gradients[1] = -10;

  ASSERT_EQ(vec.active_neurons[3], 1000);
  ASSERT_EQ(vec.activations[2], 200);
  ASSERT_EQ(vec.gradients[3], -10);
}

}  // namespace thirdai::tests
