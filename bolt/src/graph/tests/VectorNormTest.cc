
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/VectorNorm.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <memory>

namespace thirdai::bolt::tests {

const float DELTA = 0.0001;

std::vector<BoltVector> getDenseVectors() {
  BoltVector node_1_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {0.2, 0.1});
  BoltVector node_2_output =
      BoltVector::makeDenseVectorWithGradients(/* values= */ {-3.0, 2.0, 0.5});
  BoltVector node_3_output =
      BoltVector::makeDenseVectorWithGradients(/* values = */ {3.0, 4.0});

  return {node_1_output, node_2_output, node_3_output};
}

TEST(NodeUtilsTest, ComputeL1Norm) {
  auto vectors = getDenseVectors();

  ASSERT_NEAR(VectorNorm::norm(vectors[0], "l-1"), 0.3,
              /* abs_error= */ DELTA);
  ASSERT_NEAR(VectorNorm::norm(vectors[1], "l-1"), 5.5,
              /* abs_error= */ DELTA);
  ASSERT_NEAR(VectorNorm::norm(vectors[2], "l-1"), 7.0,
              /* abs_error= */ DELTA);
}

TEST(NodeUtilsTest, ComputeL2Norm) {
  auto vectors = getDenseVectors();

  ASSERT_NEAR(VectorNorm::norm(vectors[0], "euclidean"), 0.22360680566348107,
              /* abs_error= */ DELTA);
  ASSERT_NEAR(VectorNorm::norm(vectors[1], "euclidean"), 3.640054944640259,
              /* abs_error= */ DELTA);
  ASSERT_NEAR(VectorNorm::norm(vectors[2], "euclidean"), 5.0,
              /* abs_error= */ DELTA);
}

TEST(NodeUtilsTest, ComputeLInfinityNorm) {
  auto vectors = getDenseVectors();

  ASSERT_NEAR(VectorNorm::norm(vectors[0], "l-infinity"), 0.2,
              /* abs_error= */ DELTA);
  ASSERT_NEAR(VectorNorm::norm(vectors[1], "l-infinity"), 3.0,
              /* abs_error= */ DELTA);
  ASSERT_NEAR(VectorNorm::norm(vectors[2], "l-infinity"), 4.0,
              /* abs_error= */ DELTA);
}

TEST(NodeUtilsTest, ComputeNormDifference) {
  auto dense_vectors = getDenseVectors();

  auto l1_norm_value =
      VectorNorm::norm<true, true>(dense_vectors[0], dense_vectors[2], "l-1");
  auto l2_norm_value = VectorNorm::norm<true, true>(
      dense_vectors[0], dense_vectors[2], "euclidean");

  ASSERT_NEAR(l1_norm_value, 6.700, /* abs_error= */ DELTA);
  ASSERT_NEAR(l2_norm_value, 4.8010416, /* abs_error= */ DELTA);
}

}  // namespace thirdai::bolt::tests