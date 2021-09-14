#include "../../hashing/DWTA.h"
#include <gtest/gtest.h>
#include <fstream>
#include <random>
#include <set>
#include <vector>

struct TestVector {
  std::vector<uint32_t> indices;
  std::vector<float> values;
  uint32_t len;
};

float cosine_similarity(const TestVector& a, const TestVector& b,
                        bool is_dense) {
  if (is_dense) {
    float total = 0, ma = 0, mb = 0;
    for (uint32_t i = 0; i < a.len; i++) {
      total += a.values[i] * b.values[i];
      ma += a.values[i] * a.values[i];
      mb += b.values[i] * b.values[i];
    }

    return total / (std::sqrt(ma) * std::sqrt(mb));
  }
  float total = 0, ma = 0, mb = 0;
  uint32_t ia = 0, ib = 0;
  while (ia < a.len && ib < b.len) {
    if (a.indices[ia] == b.indices[ib]) {
      total += a.values[ia] * b.values[ib];
      ia++;
      ib++;
    } else if (a.indices[ia] < b.indices[ib]) {
      ia++;
    } else {
      ib++;
    }
  }
  for (uint32_t i = 0; i < a.len; i++) {
    ma += a.values[i] * a.values[i];
  }

  for (uint32_t i = 0; i < b.len; i++) {
    mb += b.values[i] * b.values[i];
  }

  return total / (std::sqrt(ma) * std::sqrt(mb));
}

std::pair<TestVector, TestVector> genRandSparseVectors(uint32_t max_dim,
                                                       float sparsity,
                                                       float sim) {
  uint32_t nonzeros = max_dim * sparsity;
  std::set<uint32_t> indices;

  std::default_random_engine rd;
  std::uniform_int_distribution<uint32_t> dist_indices(0, max_dim);
  while (indices.size() < nonzeros) {
    indices.insert(dist_indices(rd));
  }

  std::uniform_real_distribution<float> dist_vals(-10, 10);
  TestVector v1, v2;
  v1.len = nonzeros;
  v2.len = nonzeros;

  for (auto x : indices) {
    v1.indices.push_back(x);
    v2.indices.push_back(x);
    float y = dist_vals(rd);
    v1.values.push_back(y);
    v2.values.push_back(y);
  }

  std::uniform_int_distribution<uint32_t> dist_change(0, nonzeros);
  std::set<uint32_t> indices_to_change;
  uint32_t change = nonzeros * (1 - sim);
  while (indices_to_change.size() < change) {
    indices_to_change.insert(dist_change(rd));
  }

  for (auto x : indices_to_change) {
    v1.values[x] = dist_vals(rd);
    v2.values[x] = dist_vals(rd);
  }

  return {std::move(v1), std::move(v2)};
}

std::pair<TestVector, TestVector> genRandDenseVectors(uint32_t dim, float sim) {
  std::default_random_engine rd;
  std::uniform_real_distribution<float> dist_vals(-10, 10);
  TestVector v1, v2;
  v1.len = dim;
  v2.len = dim;

  for (uint32_t i = 0; i < dim; i++) {
    float x = dist_vals(rd);
    v1.indices.push_back(i);
    v2.indices.push_back(i);
    v1.values.push_back(x);
    v2.values.push_back(x);
  }

  std::uniform_int_distribution<uint32_t> dist_change(0, dim);
  std::set<uint32_t> indices_to_change;
  uint32_t change = dim * (1 - sim);
  while (indices_to_change.size() < change) {
    indices_to_change.insert(dist_change(rd));
  }

  for (auto x : indices_to_change) {
    v1.values[x] = dist_vals(rd);
    v2.values[x] = dist_vals(rd);
  }

  return {std::move(v1), std::move(v2)};
}

TEST(DWTATest, SparseHashing) {
  uint32_t dim = 10000, num_tables = 1000;
  thirdai::utils::DWTAHashFunction hash(dim, 6, num_tables, 18);

  uint32_t last = 0;
  for (uint32_t x = 5; x < 10; x++) {
    float sim = 0.1 * x;
    auto vecs = genRandSparseVectors(dim, 0.5, sim);

    uint32_t* indices[2] = {vecs.first.indices.data(),
                            vecs.second.indices.data()};
    float* values[2] = {vecs.first.values.data(), vecs.second.values.data()};
    uint32_t lens[2] = {vecs.first.len, vecs.second.len};

    uint32_t hashes[2 * num_tables];

    hash.hashSparse(2, indices, values, lens, hashes);

    uint32_t matches = 0;
    for (uint32_t i = 0; i < num_tables; i++) {
      if (hashes[i] == hashes[i + num_tables]) {
        matches++;
      }
    }

    std::cout << "Sim = " << cosine_similarity(vecs.first, vecs.second, false)
              << " matches: " << matches << std::endl;

    ASSERT_GE(matches, last);
    last = matches;
  }
  ASSERT_GE(last, 300);
}

TEST(DWTATest, DenseHashing) {
  uint32_t dim = 10000, num_tables = 1000;
  thirdai::utils::DWTAHashFunction hash(dim, 6, num_tables, 18);

  uint32_t last = 0;
  for (uint32_t x = 5; x < 10; x++) {
    float sim = 0.1 * x;
    auto vecs = genRandDenseVectors(dim, sim);

    float* values[2] = {vecs.first.values.data(), vecs.second.values.data()};
    uint32_t lens[2] = {vecs.first.len, vecs.second.len};

    uint32_t hashes[2 * num_tables];

    hash.hashDense(2, dim, values, hashes);

    uint32_t matches = 0;
    for (uint32_t i = 0; i < num_tables; i++) {
      if (hashes[i] == hashes[i + num_tables]) {
        matches++;
      }
    }

    std::cout << "Sim = " << cosine_similarity(vecs.first, vecs.second, true)
              << " matches: " << matches << std::endl;

    ASSERT_GE(matches, last);
    last = matches;
  }
  ASSERT_GE(last, 300);
}

TEST(DWTATest, DenseSparseMatch) {
  uint32_t dim = 10000, num_tables = 1000;
  thirdai::utils::DWTAHashFunction hash(dim, 6, num_tables, 18);

  for (uint32_t x = 5; x < 10; x++) {
    float sim = 0.1 * x;
    auto vecs = genRandDenseVectors(dim, sim);

    uint32_t* indices[2] = {vecs.first.indices.data(),
                            vecs.second.indices.data()};
    float* values[2] = {vecs.first.values.data(), vecs.second.values.data()};
    uint32_t lens[2] = {vecs.first.len, vecs.second.len};

    uint32_t dense_hashes[2 * num_tables];
    uint32_t sparse_hashes[2 * num_tables];

    hash.hashDense(2, dim, values, dense_hashes);

    hash.hashSparse(2, indices, values, lens, sparse_hashes);

    uint32_t matches = 0;
    for (uint32_t i = 0; i < 2 * num_tables; i++) {
      ASSERT_EQ(dense_hashes[i], sparse_hashes[i]);
    }
  }
}