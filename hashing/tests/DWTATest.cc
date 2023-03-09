#include <hashing/src/DWTA.h>
#include <hashing/tests/SparseVector.h>
#include <gtest/gtest.h>
#include <compression/src/CompressionUtils.h>
#include <cmath>
#include <random>

namespace thirdai::hashing {

using SparseVec = std::pair<std::vector<float>, std::vector<uint32_t>>;

// This test checks for a bug that causes hash values to exceed the expected
// hash range if the vector contains NaNs.
TEST(DWTATest, TestNoOutOfRangeHashWithNan) {
  std::vector<float> values = {NAN, NAN, NAN, NAN};

  uint32_t num_tables = 8, hashes_per_table = 4;
  DWTAHashFunction hash(
      /* input_dim= */ 4, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
      /* seed= */ 59302);

  std::vector<uint32_t> output_hashes(num_tables);
  hash.hashSingleDense(values.data(), values.size(), output_hashes.data());

  uint32_t range = 1 << (3 * hashes_per_table);
  for (uint32_t hash : output_hashes) {
    ASSERT_LT(hash, range);
  }
}

SparseVec sparsify_vector(std::vector<float>& dense_vec, float sparsity_level) {
  uint32_t top_k = static_cast<uint32_t>(sparsity_level * dense_vec.size());
  std::cout << "top_k: " << top_k << std::endl;
  float threshold = thirdai::compression::estimateTopKThreshold(
      dense_vec.data(), dense_vec.size(), sparsity_level, 0,
      /*sample_population_size=*/dense_vec.size());

  std::cout << "threshold: " << threshold << std::endl;
  SparseVec vec;
  for (int i = 0; i < dense_vec.size(); i++) {
    if (vec.first.size() >= top_k) {
      break;
    }
    if (std::abs(dense_vec[i]) > threshold) {
      vec.first.push_back(dense_vec[i]);
      vec.second.push_back(i);

      std::cout << "ind:" << i << " value:" << dense_vec[i] << std::endl;
    }
  }
  return vec;
}

std::vector<float> makeDenseVec(uint32_t size) {
  std::mt19937 rng;
  std::vector<float> vec(size, 0);

  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = static_cast<float>(dist(rng) / 64.0);
  }

  for (auto x : vec) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  return vec;
}

TEST(DWTATest, TestSparseDenseOverlap) {
  uint32_t size = 1000;
  float sparsity_level = 0.2;
  std::vector<float> dense_vec = makeDenseVec(size);
  SparseVec sparse_vec = sparsify_vector(dense_vec, sparsity_level);

  uint32_t num_tables = 51, hashes_per_table = 4;
  DWTAHashFunction hash(
      /* input_dim= */ size, /* _hashes_per_table= */ hashes_per_table,
      /* _num_tables= */ num_tables, /* range_pow= */ 3 * hashes_per_table,
      /* seed= */ 59302);

  std::vector<uint32_t> dense_output_hashes(num_tables),
      sparse_output_hashes(num_tables);

  hash.hashSingleDense(dense_vec.data(), dense_vec.size(),
                       dense_output_hashes.data());

  hash.hashSingleSparse(sparse_vec.second.data(), sparse_vec.first.data(),
                        sparse_vec.second.size(), sparse_output_hashes.data());

  std::cout << "sparse vec dim: " << sparse_vec.second.size() << std::endl;

  uint32_t overlaps = 0;
  for (int i = 0; i < num_tables; i++) {
    if (sparse_output_hashes[i] == dense_output_hashes[i]) {
      std::cout << sparse_output_hashes[i] << " " << dense_output_hashes[i]
                << std::endl;
      overlaps++;
    }
  }

  std::cout << "overlaps: " << overlaps
            << " ratio: " << static_cast<float>(overlaps) / num_tables
            << std::endl;
}
}  // namespace thirdai::hashing