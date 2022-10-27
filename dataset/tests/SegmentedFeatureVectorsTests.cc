#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <_types/_uint32_t.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <sys/types.h>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using thirdai::dataset::SegmentedDenseFeatureVector;
using thirdai::dataset::SegmentedFeatureVector;
using thirdai::dataset::SegmentedSparseFeatureVector;

namespace thirdai::dataset {

struct VectorSegment {
  uint32_t dim;
  bool dense;
  std::vector<uint32_t> indices;
  std::vector<float> values;
};

class SegmentedFeatureVectorTest : public testing::Test {
 protected:
  /**
   * Helper function to expose addFeatureSegment() (protected method)
   * functionality to tests.
   */
  static void addVectorFeature(SegmentedFeatureVector& vec, uint32_t dim) {
    vec.addFeatureSegment(dim);
  }

  /**
   * Helper function to expect errors concisely.
   */
  template <typename OPERATION_T>
  static void expectThrow(OPERATION_T this_shall_throw,
                          const std::string& expected_error_msg) {
    // Initially used EXPECT_THROW but linter keeps complaining.
    // Tried NOLINT but didn't work.
    try {
      this_shall_throw();
    } catch (const std::invalid_argument& e) {
      // and this tests that it has the correct message
      EXPECT_STREQ(expected_error_msg.c_str(), e.what());
    }
  }

  /**
   * Helper function to make random sparse vector segments.
   */
  static std::vector<VectorSegment> makeRandomSparseVectorSegments(
      uint32_t n_segments) {
    std::vector<VectorSegment> segments(n_segments);

    for (auto& seg : segments) {
      uint32_t dim = std::rand() % 100000;
      uint32_t n_nonzeros = std::rand() % 100;

      seg.dim = dim;
      seg.dense = false;
      for (uint32_t nonzero = 0; nonzero < n_nonzeros; nonzero++) {
        seg.indices.push_back(std::rand() % dim);
        // Value is a number between 0.0 and 10.0
        seg.values.push_back(static_cast<float>(std::rand() % 100) / 10.0);
      }
    }

    return segments;
  }

  /**
   * Helper function to make random dense vector segments.
   */
  static std::vector<VectorSegment> makeRandomDenseVectorSegments(
      uint32_t n_segments) {
    std::vector<VectorSegment> segments(n_segments);

    for (auto& seg : segments) {
      uint32_t dim = std::rand() % 100;

      seg.dim = dim;
      seg.dense = true;
      for (uint32_t elem = 0; elem < dim; elem++) {
        // Value is a number between 0.0 and 10.0
        seg.values.push_back(static_cast<float>(std::rand() % 100) / 10.0);
      }
    }

    return segments;
  }

  /**
   * Given a vector of segments, concatenate them and return a mapping from
   * indices to values of this concatenation.
   */
  static std::unordered_map<uint32_t, float> getExpectedIdxVals(
      std::vector<VectorSegment>& segments) {
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
    return expected_idx_vals;
  }

  // Exposes protected method to test functions.
  static auto getEntries(SegmentedFeatureVector& vec) { return vec.entries(); }

  /**
   * Given a segmented feature vector and a vector of segments that the vector
   * is expected to have, ensure that the segmented feature vector has the
   * correct elements.
   */
  static void checkSegmentedFeatureVectorHasSegments(
      SegmentedFeatureVector& vec, std::vector<VectorSegment>& segments) {
    auto expected_idx_vals = getExpectedIdxVals(segments);

    // Create mapping of actual idxs and vals.
    auto actual_idx_vals = vec.entries();

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

  /**
   * Given a bolt vector and a vector of segments that the vector is
   * expected to have, ensure that the segmented feature vector has the
   * correctelements.
   */
  static void checkBoltVectorHasSegments(BoltVector& vec,
                                         std::vector<VectorSegment>& segments) {
    auto expected_idx_vals = getExpectedIdxVals(segments);

    // Create mapping of actual idxs and vals.
    std::unordered_map<uint32_t, float> actual_idx_vals;
    if (vec.isDense()) {
      for (uint32_t i = 0; i < vec.len; i++) {
        actual_idx_vals[i] += vec.activations[i];
      }
    } else {
      for (uint32_t i = 0; i < vec.len; i++) {
        actual_idx_vals[vec.active_neurons[i]] += vec.activations[i];
      }
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
};

class SegmentedSparseFeatureVectorTest : public SegmentedFeatureVectorTest {};
class SegmentedDenseFeatureVectorTest : public SegmentedFeatureVectorTest {};

/**
 * Ensures that any one segment can only have either sparse features
 * or dense features and not both.
 */
TEST_F(SegmentedSparseFeatureVectorTest, AddDenseAndSparseInOneSegmentThrows) {
  // Add sparse then dense
  {
    SegmentedSparseFeatureVector vec;
    addVectorFeature(vec, /* dim = */ 10);
    vec.addSparseFeatureToSegment(/* index = */ 1, /* value = */ 1.0);
    expectThrow([&]() { vec.addDenseFeatureToSegment(/* value = */ 1.0); },
                "[SegmentedSparseFeatureVector::addDenseFeatureToSegment] A "
                "block cannot "
                "add both dense and sparse features.");
  }

  // Add dense then sparse
  {
    SegmentedSparseFeatureVector vec;
    addVectorFeature(vec, /* dim = */ 10);
    vec.addDenseFeatureToSegment(/* value = */ 1.0);
    expectThrow(
        [&]() {
          vec.addSparseFeatureToSegment(/* index = */ 1, /* value = */ 1.0);
        },
        "[SegmentedSparseFeatureVector::addSparseFeatureToSegment] A block "
        "cannot "
        "add both dense and sparse features.");
  }
}

/**
 * Ensures that sparse features must be within the specified dimension.
 */
TEST_F(SegmentedSparseFeatureVectorTest, AddSparseIndexTooHighThrows) {
  SegmentedSparseFeatureVector vec;
  addVectorFeature(vec, /* dim = */ 10);
  expectThrow(
      [&]() {
        vec.addSparseFeatureToSegment(/* index = */ 10, /* value = */ 1.0);
      },
      "[SegmentedSparseFeatureVector::addSparseFeatureToSegment] Setting value "
      "at index = 10 of vector segment with dim = 10");
}

/**
 * Ensures that segments are appropriately added to a vector.
 */
TEST_F(SegmentedSparseFeatureVectorTest,
       ProducesBoltVectorWithCorrectFeatures) {
  SegmentedSparseFeatureVector vec;

  // Add both sparse and dense segments
  auto sparse_segments = makeRandomSparseVectorSegments(3);
  auto dense_segments = makeRandomDenseVectorSegments(5);
  auto more_sparse_segments = makeRandomSparseVectorSegments(2);

  // Put together
  std::vector<VectorSegment> all_segments;
  all_segments.insert(all_segments.end(), sparse_segments.begin(),
                      sparse_segments.end());
  all_segments.insert(all_segments.end(), dense_segments.begin(),
                      dense_segments.end());
  all_segments.insert(all_segments.end(), more_sparse_segments.begin(),
                      more_sparse_segments.end());

  // Add each segment to the vector.
  for (const auto& seg : all_segments) {
    addVectorFeature(vec, seg.dim);
    if (seg.dense) {
      for (const auto& val : seg.values) {
        vec.addDenseFeatureToSegment(val);
      }
    } else {
      for (uint32_t i = 0; i < seg.indices.size(); i++) {
        vec.addSparseFeatureToSegment(seg.indices[i], seg.values[i]);
      }
    }
  }

  // Check.
  checkSegmentedFeatureVectorHasSegments(vec, all_segments);
  auto bolt_vec = vec.toBoltVector();
  checkBoltVectorHasSegments(bolt_vec, all_segments);
}

/**
 * Test that a bolt vector can be properly concatenated with an existing sparse
 * feature vector. Specifically, we want to check that the resulting vector is
 * correct when there are segments before and after the bolt vector segment,
 * and that this works with both sparse and dense bolt vectors.
 */
TEST_F(SegmentedSparseFeatureVectorTest, ExtendWithBoltVectorWorksProperly) {
  SegmentedSparseFeatureVector vec;

  uint32_t first_segment_dim = 10;
  uint32_t first_segment_idx = 3;
  float first_segment_val = 1.0;
  addVectorFeature(vec, first_segment_dim);
  vec.addSparseFeatureToSegment(first_segment_idx, first_segment_val);

  uint32_t sparse_vector_dim = 30;
  std::vector<uint32_t> sparse_vector_indices = {1, 5};
  std::vector<float> sparse_vector_values = {0.5, 5.0};
  auto sparse_vector_segment =
      BoltVector::makeSparseVector(sparse_vector_indices, sparse_vector_values);
  addVectorFeature(vec, sparse_vector_dim);
  vec.extendWithBoltVector(sparse_vector_segment);

  uint32_t dense_vector_dim = 3;
  std::vector<float> dense_vector_values = {1.0, 2.0, 3.0};
  auto dense_vector_segment = BoltVector::makeDenseVector(dense_vector_values);
  addVectorFeature(vec, dense_vector_dim);
  vec.extendWithBoltVector(dense_vector_segment);

  uint32_t last_segment_dim = 15;
  uint32_t last_segment_idx = 10;
  float last_segment_val = 0.2;
  addVectorFeature(vec, last_segment_dim);
  vec.addSparseFeatureToSegment(last_segment_idx, last_segment_val);

  auto entries = getEntries(vec);
  ASSERT_EQ(entries.size(), 7);

  uint32_t expected_offset = 0;
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 3), 1.0);
  expected_offset += first_segment_dim;
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 1), 0.5);
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 5), 5.0);
  expected_offset += sparse_vector_dim;
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 0), 1.0);
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 1), 2.0);
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 2), 3.0);
  expected_offset += dense_vector_dim;
  ASSERT_FLOAT_EQ(entries.at(expected_offset + 10), 0.2);
}

/**
 * Ensures that any one segment can only have either sparse features
 * or dense features and not both.
 */
TEST_F(SegmentedDenseFeatureVectorTest, AddSparseThrows) {
  SegmentedDenseFeatureVector vec;
  addVectorFeature(vec, /* dim = */ 10);
  expectThrow(
      [&]() {
        vec.addSparseFeatureToSegment(/* index = */ 1, /* value = */ 1.0);
      },
      "[SegmentedDenseFeatureVector::addSparseFeatureToSegment] "
      "SegmentedDenseFeatureVector does not accept sparse features.");
}

/**
 * Ensures a dense segment does not accept more values than the
 * specified dimension.
 */
TEST_F(SegmentedDenseFeatureVectorTest, AddTooManyValuesThrows) {
  SegmentedDenseFeatureVector vec;
  addVectorFeature(vec, /* dim = */ 1);
  vec.addDenseFeatureToSegment(/* value = */ 1.0);
  expectThrow([&]() { vec.addDenseFeatureToSegment(/* value = */ 1.0); },
              "[SegmentedDenseFeatureVector::addDenseFeatureToSegment] Adding "
              "2-th dense feature to vector segment with dim = 1");
}

/**
 * Ensures the number of dense features in each segment
 * is no less than the specified dimenion.
 */
TEST_F(SegmentedDenseFeatureVectorTest, PrematurelyAddingNewSegmentThrows) {
  SegmentedDenseFeatureVector vec;
  addVectorFeature(vec, /* dim = */ 10);
  vec.addDenseFeatureToSegment(/* value = */ 1.0);
  expectThrow([&]() { addVectorFeature(vec, /* dim = */ 10); },
              "[SegmentedDenseFeatureVector::addFeatureSegment] Adding vector "
              "segment before "
              "completing previous segment. Previous segment expected to "
              "have dim = 10 but only 1 dense features were added.");
}

/**
 * Ensures that the segments are appropriately added to vector.
 */
TEST_F(SegmentedDenseFeatureVectorTest, ProducesBoltVectorWithCorrectFeatures) {
  SegmentedDenseFeatureVector vec;

  // Make and add segments
  auto segments = makeRandomDenseVectorSegments(1);
  for (const auto& seg : segments) {
    addVectorFeature(vec, seg.dim);
    for (const auto& val : seg.values) {
      vec.addDenseFeatureToSegment(val);
    }
  }

  // Check.
  checkSegmentedFeatureVectorHasSegments(vec, segments);
  auto bolt_vec = vec.toBoltVector();
  checkBoltVectorHasSegments(bolt_vec, segments);
}

/**
 * Test that a bolt vector can be properly concatenated with an existing dense
 * feature vector. Specifically, we want to check that the resulting vector is
 * correct when there are segments before and after the bolt vector segment.
 * Unlike SegmentedSparseFeatureVector, SegmentedDenseFeatureVector cannot be
 * extended with a bolt vector.
 */
TEST_F(SegmentedDenseFeatureVectorTest, ExtendWithBoltVectorWorksProperly) {
  SegmentedSparseFeatureVector vec;

  uint32_t first_segment_dim = 10;
  std::vector<float> first_segment_vals(first_segment_dim);
  std::iota(first_segment_vals.begin(), first_segment_vals.end(), 0.0);
  addVectorFeature(vec, first_segment_dim);
  for (auto val : first_segment_vals) {
    vec.addDenseFeatureToSegment(val);
  }

  uint32_t dense_vector_dim = 3;
  std::vector<float> dense_vector_values = {1.0, 2.0, 3.0};
  auto dense_vector_segment = BoltVector::makeDenseVector(dense_vector_values);
  addVectorFeature(vec, dense_vector_dim);
  vec.extendWithBoltVector(dense_vector_segment);

  uint32_t last_segment_dim = 10;
  std::vector<float> last_segment_vals(last_segment_dim);
  std::iota(last_segment_vals.begin(), last_segment_vals.end(), 0.0);
  addVectorFeature(vec, last_segment_dim);
  for (auto val : last_segment_vals) {
    vec.addDenseFeatureToSegment(val);
  }

  auto final_vector = vec.toBoltVector();

  for (uint32_t i = 0; i < first_segment_dim; i++) {
    ASSERT_EQ(final_vector.activations[i], first_segment_vals[i]);
  }

  for (uint32_t i = 0; i < dense_vector_dim; i++) {
    ASSERT_EQ(final_vector.activations[first_segment_dim + i],
              dense_vector_values[i]);
  }

  for (uint32_t i = 0; i < last_segment_dim; i++) {
    ASSERT_EQ(
        final_vector.activations[first_segment_dim + dense_vector_dim + i],
        last_segment_vals[i]);
  }
}

TEST_F(SegmentedDenseFeatureVectorTest, ExtendWithSparseBoltVectorThrows) {
  SegmentedDenseFeatureVector vec;

  auto sparse_vector_segment =
      BoltVector::singleElementSparseVector(/* active_neuron= */ 5);

  addVectorFeature(vec, /* dim= */ 10);

  expectThrow(
      [&]() { vec.extendWithBoltVector(sparse_vector_segment); },
      "[SegmentedDenseFeatureVector::extendWithBoltVector] "
      "SegmentedDenseFeatureVector cannot be extended with a sparse bolt "
      "vector.");
}
}  // namespace thirdai::dataset
