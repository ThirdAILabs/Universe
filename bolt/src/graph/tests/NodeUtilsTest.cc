
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/NodeUtils.h>
#include <bolt/src/graph/tests/MockNode.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <memory>
#include <random>

namespace thirdai::bolt::tests {

const uint32_t len = 10;

BoltVector getVectorWithRandomActivations(uint32_t length) {
  std::vector<float> values(length);
  std::vector<uint32_t> active_neurons(length);

  std::default_random_engine generator;
  static std::uniform_real_distribution<float> distribution(-5.0, 5.0);

  for (uint32_t index = 0; index < length; index++) {
    active_neurons.push_back(index);
    float random_activation = distribution(generator);
    values.push_back(random_activation);
  }
  return BoltVector(&active_neurons[0], &values[0], nullptr, length);
}

std::vector<NodePtr> getNodesForTesting(
    const std::vector<uint32_t>& input_dense_dims,
    const std::vector<BoltVector>& input_vectors) {
  std::vector<NodePtr> nodes;
  for (uint32_t input_index = 0; input_index < input_dense_dims.size();
       input_index++) {
    uint32_t input_dense_dim = input_dense_dims[input_index];
    const BoltVector& input_vector = input_vectors[input_index];

    nodes.emplace_back(
        std::make_shared<MockNodeWithOutput>(input_vector, input_dense_dim));
  }
  return nodes;
}

void testVectorNormComputation(const std::vector<uint32_t>& input_dense_dims,
                               const std::vector<BoltVector>& vectors,
                               const std::string& norm_order) {
  auto nodes = getNodesForTesting(input_dense_dims, vectors);
  for (uint32_t node_index = 0; node_index < nodes.size(); node_index++) {
    auto& output =
        nodes[node_index]->getOutputVector(/* vec_index= */ node_index);

    double input_node_norm = NodeProperties::norm(output, norm_order);

    double expected_norm = 0.0;
    if (getNorm(norm_order) == LPNorm::L1) {
      for (uint32_t activation_index = 0; activation_index < len;
           activation_index++) {
        expected_norm += abs(output.activations[activation_index]);
      }

    } else if (getNorm(norm_order) == LPNorm::Euclidean) {
      for (uint32_t activation_index = 0; activation_index < len;
           activation_index++) {
        expected_norm += pow(output.activations[activation_index], 2.0);
      }
      expected_norm = sqrt(expected_norm);

    } else if (getNorm(norm_order) == LPNorm::LInfinity) {
      expected_norm = static_cast<double>(abs(*output.activations));
      for (uint32_t activation_index = 0; activation_index < len;
           activation_index++) {
        expected_norm = std::max(
            expected_norm,
            abs(static_cast<double>(output.activations[activation_index])));
      }
    }
    ASSERT_DOUBLE_EQ(input_node_norm, expected_norm);
  }
}

TEST(NodeUtilsTest, ComputeL1PNorm) {
  BoltVector node_1_output = getVectorWithRandomActivations(len);
  BoltVector node_2_output = getVectorWithRandomActivations(len);

  testVectorNormComputation(
      /* input_dense_dims= */ std::vector<uint32_t>(2, len),
      /* vectors =*/{node_1_output, node_2_output},
      /* norm_order= */ "l-1");
}

TEST(NodeUtilsTest, ComputeL2Norm) {
  BoltVector node_1_output = getVectorWithRandomActivations(len);
  BoltVector node_2_output = getVectorWithRandomActivations(len);
  testVectorNormComputation(
      /* input_dense_dims= */ std::vector<uint32_t>(2, len),
      /* vectors =*/{node_1_output, node_2_output},
      /* norm_order= */ "euclidean");
}

TEST(NodeUtilsTest, ComputeLInfinityNorm) {
  BoltVector node_1_output = getVectorWithRandomActivations(len);
  BoltVector node_2_output = getVectorWithRandomActivations(len);
  testVectorNormComputation(
      /* input_dense_dims= */ std::vector<uint32_t>(2, len),
      /* vectors =*/{node_1_output, node_2_output},
      /* norm_order= */ "l-infinity");
}

}  // namespace thirdai::bolt::tests