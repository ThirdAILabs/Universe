#include "CLI11.hpp"
#include <bolt/src/utils/CompressedVector.h>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

namespace thirdai::bolt::cli {

struct Options {
  uint64_t uncompressed_size = 100000;
  uint64_t compressed_size = 10000;
  uint64_t block_size = 64;
  float mean = 0.0f;
  float stddev = 1.0f;
  uint64_t step_size = 10000;
  bool use_sign_bit = false;
};

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

  error = std::sqrt(error) / static_cast<float>(num_elements);
  return error;
}

void runReconstructionAnalysis(const Options& options) {
  // Because we love the answer to life, universe and everything.
  const uint32_t seed = 42;
  using ElementType = float;

  for (uint64_t compressed_size = options.compressed_size;
       compressed_size <= options.uncompressed_size;
       compressed_size += options.step_size) {
    std::vector<ElementType> uncompressed_vector(options.uncompressed_size);

    // Random number generator. Reuse the seed.
    std::mt19937_64 gen64(seed);
    std::normal_distribution<> normal_distribution{/*mean=*/options.mean,
                                                   /*variance=*/options.stddev};
    auto generator = [&gen64, &normal_distribution]() {
      return normal_distribution(gen64);
    };

    std::generate(uncompressed_vector.begin(), uncompressed_vector.end(),
                  generator);

    CompressedVector<ElementType> compressed_vector(
        uncompressed_vector, compressed_size, options.block_size, seed,
        options.use_sign_bit);

    float error = single_vector_reconstruction_error(compressed_vector,
                                                     uncompressed_vector);
    std::cout << "Reconstruction error@compressed_size(" << compressed_size
              << "): " << error << std::endl;
  }
}

}  // namespace thirdai::bolt::cli

int main(int argc, char** argv) {
  CLI::App app{"Command line driver app, without Python"};

  thirdai::bolt::cli::Options options;
  // clang-format off
  app.add_option("-N,--uncompressed-size", options.uncompressed_size, "Uncompressed size");
  app.add_option("-m,--compressed-size", options.compressed_size, "compressed size");
  app.add_option("-d,--step-size", options.step_size, "");
  app.add_option("-b,--block-size", options.block_size, "");
  app.add_option("--use-sign-bit", options.use_sign_bit, "");
  // clang-format on

  CLI11_PARSE(app, argc, argv);
  thirdai::bolt::cli::runReconstructionAnalysis(options);
  return 0;
}
