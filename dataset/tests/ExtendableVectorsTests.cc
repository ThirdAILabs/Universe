#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/ExtendableVectors.h>
#include <sys/types.h>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using thirdai::dataset::DenseExtendableVector;
using thirdai::dataset::ExtendableVector;
using thirdai::dataset::SparseExtendableVector;

namespace thirdai::dataset {

struct VectorSegment {
  uint32_t dim;
  bool dense;
  std::vector<uint32_t> indices;
  std::vector<float> values;
};

class ExtendableVectorTest : public testing::Test {
 protected:
  /**
   * Helper function to expose extendByDim() (protected method)
   * functionality to tests.
   */
  static void extendVector(ExtendableVector& vec, uint32_t dim) {
    vec.extendByDim(dim);
  }

  /**
   * Helper function to expect errors concisely.
   */
  template <typename OPERATION_T>
  static void expectThrow(OPERATION_T this_shall_throw,
                          const std::string& expected_error_msg) {
    EXPECT_THROW(
        {
          try {
            this_shall_throw();
          } catch (const std::invalid_argument& e) {
            // and this tests that it has the correct message
            EXPECT_STREQ(expected_error_msg.c_str(), e.what());
            throw;
          }
        },
        std::invalid_argument);
  }

  static std::vector<VectorSegment> makeRandomSparseVectorSegments(uint32_t n_segments) {
    std::vector<VectorSegment> segments(n_segments);

    for (auto& seg : segments) {
      uint32_t dim = random() % 100000;
      uint32_t n_nonzeros = random() % 100;

      seg.dim = dim;
      seg.dense = false;
      for (uint32_t nonzero = 0; nonzero < n_nonzeros; nonzero++) {
        seg.indices.push_back(random() % dim);
        seg.values.push_back(static_cast<float>(random() % 100) / 10.0);
      }
    }
    
    return segments;
  }

  static void checkVectorHasSegments(ExtendableVector& vec, std::vector<VectorSegment>& segments) {
    // Create mapping of expected idxs and vals.
    std::unordered_map<uint32_t, float> expected_idx_vals;
    uint32_t seg_start_idx = 0;
    for (const auto& seg : segments) {
      if (seg.dense) {
        for (uint32_t i = 0; i < seg.values.size(); i++) {
          expected_idx_vals[seg_start_idx + i] = seg.values[i];
        }
      } else {
        for (uint32_t i = 0; i < seg.indices.size(); i++) {
          expected_idx_vals[seg_start_idx + seg.indices[i]] += seg.values[i];
        }
      }
      seg_start_idx += seg.dim;
    }

    // Create mapping of actual idxs and vals.
    std::unordered_map<uint32_t, float> actual_idx_vals;
    for (const auto& [idx, val] : vec.entries()) {
      actual_idx_vals[idx] += val;
    }

    // Check all values in actual_idx_vals are as expected.
    for (const auto& [idx, val] : expected_idx_vals) {
      ASSERT_EQ(val, actual_idx_vals[idx]);
    }

    // Check that all index-value pairs in actual_idx_vals
    // are supposed to be there.
    for (const auto& [idx, val] : actual_idx_vals) {
      ASSERT_EQ(val, expected_idx_vals[idx]);
    }
  }

  static std::vector<VectorSegment> makeRandomDenseVectorSegments(uint32_t n_segments) {
    std::vector<VectorSegment> segments(n_segments);

    for (auto& seg : segments) {
      uint32_t dim = random() % 100;
      
      seg.dim = dim;
      seg.dense = true;
      for (uint32_t elem = 0; elem < dim; elem++) {
        seg.values.push_back(static_cast<float>(random() % 100) / 10.0);
      }
    }
    
    return segments;
  }
};

class SparseExtendableVectorTest : public ExtendableVectorTest {};
class DenseExtendableVectorTest : public ExtendableVectorTest {};

/**
 * Ensures that any one extension can only have either sparse features
 * or dense features and not both.
 */
TEST_F(SparseExtendableVectorTest, AddDenseAndSparseInOneExtensionThrows) {
  // Add sparse then dense
  {
    SparseExtendableVector vec;
    extendVector(vec, /* dim = */ 10);
    vec.addExtensionSparseFeature(/* index = */ 1, /* value = */ 1.0);
    expectThrow(
      [&]() { vec.addExtensionDenseFeature(/* value = */ 1.0); },
      "[SparseExtendableVector::addExtensionSparseFeature] A block cannot "
      "add both dense and sparse features.");
  }

  // Add dense then sparse
  {
    SparseExtendableVector vec;
    extendVector(vec, /* dim = */ 10);
    vec.addExtensionDenseFeature(/* value = */ 1.0);
    expectThrow(
      [&]() {
        vec.addExtensionSparseFeature(/* index = */ 1, /* value = */ 1.0);
      },
      "[SparseExtendableVector::addExtensionSparseFeature] A block cannot "
      "add both dense and sparse features.");
  }
}

TEST_F(SparseExtendableVectorTest, AddSparseIndexTooHighThrows) {
  SparseExtendableVector vec;
  extendVector(vec, /* dim = */ 10);
  expectThrow(
    [&]() {
      vec.addExtensionSparseFeature(/* index = */ 10, /* value = */ 1.0);
    },
    "[SparseExtendableVector::addExtensionSparseFeature] Setting value "
    "at index = 10 of extension vector with dim = 10");
}

TEST_F(SparseExtendableVectorTest, ProducesBoltVectorWithCorrectFeatures) {
  SparseExtendableVector vec;
  
  auto sparse_segments = makeRandomSparseVectorSegments(3);
  auto dense_segments = makeRandomDenseVectorSegments(5);
  auto more_sparse_segments = makeRandomSparseVectorSegments(2);

  std::vector<VectorSegment> all_segments;
  all_segments.insert(all_segments.end(), sparse_segments.begin(), sparse_segments.end());
  all_segments.insert(all_segments.end(), dense_segments.begin(), dense_segments.end());
  all_segments.insert(all_segments.end(), more_sparse_segments.begin(), more_sparse_segments.end());


  for (const auto& seg : all_segments) {
    extendVector(vec, seg.dim);
    if (seg.dense) {
      for (const auto& val : seg.values) {
        vec.addExtensionDenseFeature(val);
      }
    } else {
      for (uint32_t i = 0; i < seg.indices.size(); i++) {
        vec.addExtensionSparseFeature(seg.indices[i], seg.values[i]);
      }
    }
  }

  checkVectorHasSegments(vec, all_segments); 
}

/**
 * Ensures that any one extension can only have either sparse features
 * or dense features and not both.
 */
TEST_F(DenseExtendableVectorTest, AddSparseThrows) {
  DenseExtendableVector vec;
  extendVector(vec, /* dim = */ 10);
  expectThrow(
    [&]() { vec.addExtensionSparseFeature(/* index = */ 1, /* value = */ 1.0); },
    "[DenseExtendableVector::addExtensionSparseFeature] "
    "DenseExtendableVector does not accept sparse features.");
}

TEST_F(DenseExtendableVectorTest, AddTooManyValuesThrows) {
  DenseExtendableVector vec;
  extendVector(vec, /* dim = */ 1);
  vec.addExtensionDenseFeature(/* value = */ 1.0);
  expectThrow(
    [&]() {
      vec.addExtensionDenseFeature(/* value = */ 1.0);
    },
    "[DenseExtendableVector::addExtensionDenseFeature] Adding "
    "2-th dense feature to extension vector with dim = 1");
}

TEST_F(DenseExtendableVectorTest, PrematureExtensionThrows) {
  DenseExtendableVector vec;
  extendVector(vec, /* dim = */ 10);
  vec.addExtensionDenseFeature(/* value = */ 1.0);
  expectThrow(
    [&]() {
      extendVector(vec, /* dim = */ 10);
    },
    "[DenseExtendableVector::extendByDim] Extending vector before "
    "completing previous extension. Previous extension expected to "
    "have dim = 10 but only 1 dense features were added.");
}

TEST_F(DenseExtendableVectorTest, ProducesBoltVectorWithCorrectFeatures) {
  DenseExtendableVector vec;
  
  auto segments = makeRandomDenseVectorSegments(10);

  for (const auto& seg : segments) {
    extendVector(vec, seg.dim);
    for (const auto& val : seg.values) {
      vec.addExtensionDenseFeature(val);
    }
  }

  checkVectorHasSegments(vec, segments); 
}

}  // namespace thirdai::dataset
