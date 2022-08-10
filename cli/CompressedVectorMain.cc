#include "CLI11.hpp"
#include <gtest/gtest.h>
#include <compression/src/CompressedVector.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

namespace thirdai::bolt::cli {

struct Options {
  uint64_t uncompressed_size = 100000;
  uint64_t compressed_size = 10000;
  uint64_t block_size = 1;
  float mean = 0.0;
  float stddev = 2.0;
  uint64_t step_size = 10000;
  bool use_sign_bit = false;
};

template <class ELEMENT_TYPE>
float single_vector_reconstruction_error(
    const CompressedVector<ELEMENT_TYPE>* compressed_vector,
    const std::vector<ELEMENT_TYPE>& large_vector) {
  float error = 0;
  for (size_t i = 0; i < large_vector.size(); i++) {
    float diff = large_vector[i] - compressed_vector->get(i);
    error += diff * diff;  // Squared error.
  }

  error = std::sqrt(error) / static_cast<float>(large_vector.size());
  return error;
}

void runReconstructionAnalysis(const Options& options) {
  // Because we love the answer to life, universe and everything.
  const uint32_t seed = 42;
  using ElementType = float;

  // Create a normal vector.
  std::vector<ElementType> uncompressed_vector(options.uncompressed_size);

  // Populate the vector with values from a normal distribution.
  std::mt19937_64 gen64(seed);
  std::normal_distribution<> normal_distribution{/*mean=*/options.mean,
                                                 /*variance=*/options.stddev};
  auto generator = [&gen64, &normal_distribution]() {
    return normal_distribution(gen64);
  };

  std::generate(uncompressed_vector.begin(), uncompressed_vector.end(),
                generator);

  // Construction to make a CompressedVector parameterized by size.
  // The remaining parameters are captured by references.
  auto make_compressed_vector = [&](uint64_t compressed_size) {
    std::unique_ptr<CompressedVector<float>> cv{nullptr};
    if (options.use_sign_bit) {
      cv = std::make_unique<UnbiasedSketch<float>>(
          uncompressed_vector, compressed_size, options.block_size, seed);
    } else {
      cv = std::make_unique<BiasedSketch<float>>(
          uncompressed_vector, compressed_size, options.block_size, seed);
    }
    return cv;
  };

  // Vary compressed_size from a lower limit all the way to uncompressed_size
  uint64_t upper_bound =
      options.uncompressed_size * std::log(options.uncompressed_size);
  for (uint64_t compressed_size = options.compressed_size;
       compressed_size <= upper_bound; compressed_size += options.step_size) {
    std::unique_ptr<CompressedVector<ElementType>> compressed_vector =
        make_compressed_vector(compressed_size);

    float error = single_vector_reconstruction_error(compressed_vector.get(),
                                                     uncompressed_vector);
    std::cout << "Reconstruction Error | "
              << (options.use_sign_bit ? "Unbiased " : "Biased")
              << " sketch: (N: " << options.uncompressed_size
              << ", m: " << compressed_size << "): " << error << std::endl;
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
  app.add_option("--mu", options.mean, "");
  app.add_option("--std", options.stddev, "");
  app.add_option("-b,--block-size", options.block_size, "");
  app.add_flag("--use-sign-bit", options.use_sign_bit, "");
  // clang-format on

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    exit(app.exit(e));
  }
  try {
    thirdai::bolt::cli::runReconstructionAnalysis(options);
  } catch (...) {
    std::cerr << "CatchAll failed from bolt - shut up, clang!" << std::endl;
  }
  return 0;
}
