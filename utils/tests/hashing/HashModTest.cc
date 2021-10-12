#include "../../hashing/HashMod.h"
#include "../../hashing/HashModPow2.h"
#include "../../hashing/SRP.h"
#include "CosineSim.h"
#include <gtest/gtest.h>

using thirdai::utils::HashMod;
using thirdai::utils::HashModPow2;
using thirdai::utils::SparseRandomProjection;
using thirdai::utils::lsh_testing::CosineSim;

static constexpr uint32_t seed = 24908942;
static constexpr uint32_t iters = 10000;
static constexpr uint32_t vec_dim = 100;
static constexpr uint32_t num_tables = 10;
static constexpr uint32_t hashes_per_table = 20;

void runDenseHashModTest(thirdai::utils::HashFunction& hash_func,
                         uint32_t range) {
  CosineSim vec_gen(seed);

  uint32_t* hashes = new uint32_t[num_tables];
  for (uint32_t i = 0; i < iters; i++) {
    auto rand_vecs = vec_gen.getRandomDenseVectors(0.8, vec_dim);

    hash_func.hashSingleDense(rand_vecs.v1.values, rand_vecs.v1.dim, hashes);
    for (uint32_t j = 0; j < num_tables; j++) {
      ASSERT_LT(hashes[j], range);
    }

    hash_func.hashSingleDense(rand_vecs.v2.values, rand_vecs.v2.dim, hashes);
    for (uint32_t j = 0; j < num_tables; j++) {
      ASSERT_LT(hashes[j], range);
    }
  }

  delete[] hashes;
}

void runSparseHashModTest(thirdai::utils::HashFunction& hash_func,
                          uint32_t range) {
  CosineSim vec_gen(seed);

  uint32_t* hashes = new uint32_t[num_tables];
  for (uint32_t i = 0; i < iters; i++) {
    auto rand_vecs = vec_gen.getRandomSparseVectors(0.8, vec_dim / 2, vec_dim);

    hash_func.hashSingleSparse(rand_vecs.v1.indices, rand_vecs.v1.values,
                               rand_vecs.v1.len, hashes);
    for (uint32_t j = 0; j < num_tables; j++) {
      ASSERT_LT(hashes[j], range);
    }

    hash_func.hashSingleSparse(rand_vecs.v2.indices, rand_vecs.v2.values,
                               rand_vecs.v2.len, hashes);
    for (uint32_t j = 0; j < num_tables; j++) {
      ASSERT_LT(hashes[j], range);
    }
  }

  delete[] hashes;
}

TEST(HashModTests, HashModDenseVector) {
  uint32_t mod = 100;
  HashMod<SparseRandomProjection> hash_func(mod, vec_dim, hashes_per_table,
                                            num_tables, seed);
  runDenseHashModTest(hash_func, mod);
}

TEST(HashModTests, HashModSparseVector) {
  uint32_t mod = 100;
  HashMod<SparseRandomProjection> hash_func(mod, vec_dim, hashes_per_table,
                                            num_tables, seed);
  runSparseHashModTest(hash_func, mod);
}

TEST(HashModTests, HashModPow2DenseVector) {
  uint32_t output_bits = 7;
  HashModPow2<SparseRandomProjection> hash_func(
      output_bits, vec_dim, hashes_per_table, num_tables, seed);

  runDenseHashModTest(hash_func, 1 << output_bits);
}

TEST(HashModTests, HashModPow2SparseVector) {
  uint32_t output_bits = 7;
  HashModPow2<SparseRandomProjection> hash_func(
      output_bits, vec_dim, hashes_per_table, num_tables, seed);

  runSparseHashModTest(hash_func, 1 << output_bits);
}
