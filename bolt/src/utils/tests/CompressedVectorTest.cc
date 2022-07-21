#include <bolt/src/utils/CompressedVector.h>
#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <random>

namespace thirdai::bolt::tests {

template <class ELEMENT_TYPE>
float reconstruction_error(
    const CompressedVector<ELEMENT_TYPE>& compressed_vector,
    const std::vector<ELEMENT_TYPE>& large_vector) {
  float error = 0;

  size_t num_elements = large_vector.size();
  for (size_t i = 0; i < num_elements; i++) {
    float diff =
        static_cast<float>(compressed_vector[i] - compressed_vector[i]);
    error += diff * diff;  // Squared error.
  }

  error = error / static_cast<float>(num_elements);
  return error;
}

void runReconstructionTest() {
  const uint64_t uncompressed_size = 100000;
  const uint64_t compressed_size = 100;
  const uint64_t block_size = 64;

  // Because we love the answer to life, universe and everything.
  const size_t seed = 42;

  std::vector<uint64_t> uncompressed_vector(uncompressed_size);

  // Random number generator. Reuse the seed.
  std::mt19937_64 gen64(seed);
  std::generate(uncompressed_vector.begin(), uncompressed_vector.end(), gen64);

  CompressedVector<uint64_t> compressed_vector(
      uncompressed_vector, compressed_size, block_size, seed);

  float error = reconstruction_error(compressed_vector, uncompressed_vector);
  std::cout << "Reconstruction error: " << error << std::endl;
}

}  // namespace thirdai::bolt::tests

TEST(CompressedVectorTests, Mega) {
  thirdai::bolt::tests::runReconstructionTest();
}
