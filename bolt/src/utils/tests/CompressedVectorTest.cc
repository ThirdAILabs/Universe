#include <bolt/src/utils/CompressedVector.h>
#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <random>

namespace thirdai::bolt::tests {

template <class ELEMENT_TYPE>
float single_vector_reconstruction_error(
    const CompressedVector<ELEMENT_TYPE>& compressed_vector,
    const std::vector<ELEMENT_TYPE>& large_vector) {
  float error = 0;
  size_t num_elements = large_vector.size();
  for (size_t i = 0; i < num_elements; i++) {
    float diff = static_cast<float>(large_vector[i] - compressed_vector.get(i));
    error += diff * diff;  // Squared error.
  }

  error = std::sqrtf(error) / static_cast<float>(num_elements);
  return error;
}

void runReconstructionTest() {
  const uint64_t uncompressed_size = static_cast<uint64_t>(5e6);
  const uint64_t block_size = 16;

  // Because we love the answer to life, universe and everything.
  const uint32_t seed = 42;
  using ElementType = float;

  for (uint64_t compressed_size = 10000; compressed_size <= uncompressed_size;
       compressed_size += 10000) {
    std::vector<ElementType> uncompressed_vector(uncompressed_size);

    // Random number generator. Reuse the seed.
    std::mt19937_64 gen64(seed);
    std::normal_distribution<> normal_distribution{/*mean=*/0, /*variance=*/4};
    std::uniform_real_distribution<> uniform_real_distribution{-100, 100};
    auto generator = [&gen64, &normal_distribution,
                      &uniform_real_distribution]() {
      // return normal_distribution(gen64);
      return uniform_real_distribution(gen64);
    };

    std::generate(uncompressed_vector.begin(), uncompressed_vector.end(),
                  generator);

    CompressedVector<ElementType> compressed_vector(
        uncompressed_vector, compressed_size, block_size, seed,
        /*use_sign_bit=*/false);

    float error = single_vector_reconstruction_error(compressed_vector,
                                                     uncompressed_vector);
    std::cout << "Reconstruction error@compressed_size(" << compressed_size
              << "): " << error << std::endl;
  }
}

}  // namespace thirdai::bolt::tests

TEST(CompressedVectorTests, Mega) {
  // This is not really a "test" test at the moment. @jerin-thirdai is using the
  // test to generate a binary that can be used as a driver to prototype.
  thirdai::bolt::tests::runReconstructionTest();
}
