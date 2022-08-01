#include <cereal/archives/binary.hpp>
#include <cereal/details/helpers.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <sstream>

namespace thirdai::bolt::tests {

std::stringstream serializeBoltVector(BoltVector& vector) {
  std::stringstream output_stream;
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(vector);
  return output_stream;
}

BoltVector deserializeBoltVector(std::stringstream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  BoltVector vector;
  iarchive(vector);
  return vector;
}

BoltVector makeDataOwningBoltVector(const std::vector<uint32_t>& active_neurons,
                                    const std::vector<float>& activations,
                                    const std::vector<float>& gradients) {
  BoltVector vector(activations.size(), /* is_dense= */ active_neurons.empty(),
                    /* has_gradient= */ !gradients.empty());

  if (!active_neurons.empty()) {
    EXPECT_EQ(active_neurons.size(), activations.size());
    std::copy(active_neurons.begin(), active_neurons.end(),
              vector.active_neurons);
  }
  std::copy(activations.begin(), activations.end(), vector.activations);

  if (!gradients.empty()) {
    EXPECT_EQ(gradients.size(), activations.size());
    std::copy(gradients.begin(), gradients.end(), vector.gradients);
  }

  return vector;
}

BoltVector makeDataNonOwningBoltVector(std::vector<uint32_t>& active_neurons,
                                       std::vector<float>& activations,
                                       std::vector<float>& gradients) {
  if (!active_neurons.empty()) {
    EXPECT_EQ(active_neurons.size(), activations.size());
  }
  if (!gradients.empty()) {
    EXPECT_EQ(gradients.size(), activations.size());
  }

  return BoltVector(active_neurons.data(), activations.data(), gradients.data(),
                    activations.size());
}

void assertBoltVectorsAreEqual(const BoltVector& a, const BoltVector& b) {
  EXPECT_EQ(a.len, b.len);
  EXPECT_EQ(a.isDense(), b.isDense());
  EXPECT_EQ(a.hasGradients(), b.hasGradients());

  for (uint32_t i = 0; i < a.len; i++) {
    if (!a.isDense()) {
      ASSERT_EQ(a.active_neurons[i], b.active_neurons[i]);
    }
    ASSERT_EQ(a.activations[i], b.activations[i]);
    if (a.hasGradients()) {
      ASSERT_EQ(a.gradients[i], b.gradients[i]);
    }
  }
}

void testDataOwningBoltVector(const std::vector<uint32_t>& active_neurons,
                              const std::vector<float>& activations,
                              const std::vector<float>& gradients) {
  auto vector =
      makeDataOwningBoltVector(active_neurons, activations, gradients);

  auto serialized = serializeBoltVector(vector);

  BoltVector deserialized_vector = deserializeBoltVector(serialized);

  assertBoltVectorsAreEqual(vector, deserialized_vector);
}

void testDataNonOwningBoltVector(std::vector<uint32_t>& active_neurons,
                                 std::vector<float>& activations,
                                 std::vector<float>& gradients) {
  auto vector =
      makeDataNonOwningBoltVector(active_neurons, activations, gradients);

  auto serialized = serializeBoltVector(vector);

  BoltVector deserialized_vector = deserializeBoltVector(serialized);

  assertBoltVectorsAreEqual(vector, deserialized_vector);
}

TEST(BoltVectorSerialization, SaveLoadSparseBoltVectorWithGradients) {
  std::vector<uint32_t> active_neurons = {7, 2, 9, 11, 32, 44, 65};
  std::vector<float> activations = {4.25, -0.625, 1.5, 11.0, -2.75, 3.125, 0.5};
  std::vector<float> gradients = {0.375, -1.0, 0.0, 2.875, 3.25, -0.25, 7.25};

  testDataOwningBoltVector(active_neurons, activations, gradients);
  testDataNonOwningBoltVector(active_neurons, activations, gradients);
}

TEST(BoltVectorSerialization, SaveLoadSparseBoltVectorWithoutGradients) {
  std::vector<uint32_t> active_neurons = {7, 2, 9, 11, 32, 44, 65};
  std::vector<float> activations = {4.25, -0.625, 1.5, 11.0, -2.75, 3.125, 0.5};
  std::vector<float> gradients = {};

  testDataOwningBoltVector(active_neurons, activations, gradients);
  testDataNonOwningBoltVector(active_neurons, activations, gradients);
}

TEST(BoltVectorSerialization, SaveLoadDenseBoltVectorWithGradients) {
  std::vector<uint32_t> active_neurons = {};
  std::vector<float> activations = {4.25, -0.625, 1.5, 11.0, -2.75, 3.125, 0.5};
  std::vector<float> gradients = {0.375, -1.0, 0.0, 2.875, 3.25, -0.25, 7.25};

  testDataOwningBoltVector(active_neurons, activations, gradients);
  testDataNonOwningBoltVector(active_neurons, activations, gradients);
}

TEST(BoltVectorSerialization, SaveLoadDenseBoltVectorWithoutGradients) {
  std::vector<uint32_t> active_neurons = {};
  std::vector<float> activations = {4.25, -0.625, 1.5, 11.0, -2.75, 3.125, 0.5};
  std::vector<float> gradients = {};

  testDataOwningBoltVector(active_neurons, activations, gradients);
  testDataNonOwningBoltVector(active_neurons, activations, gradients);
}

}  // namespace thirdai::bolt::tests