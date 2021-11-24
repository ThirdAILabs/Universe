#pragma once

#include "Similarity.h"
#include <hashing/src/HashFunction.h>
#include <random>
#include <set>
#include <vector>

namespace thirdai::hashing {

/**
 * This function takes in a hash function and a similarity function and runs
 * num_tests number of tests with different pairs of vectors with increasing
 * uniformly distributed similarity values from 0 to 1, and asserts that the
 * difference in the calculated and measured similarity values are all smaller
 * in absolute value than max_diff, and finally checks that the average
 * difference is smaller in absolute value than max_avg_diff. Finally, sparsity
 * is the percent of non zeros we want to have in our test vectors.
 */
void runSparseSimilarityTest(const thirdai::hashing::HashFunction& hash,
                             Similarity& sim, uint32_t dim, uint32_t num_tables,
                             uint32_t num_tests, float sparsity, float max_diff,
                             float max_avg_diff);

/** This is the same as the runSparseTest, except it occurs on dense vectors */
void runDenseSimilarityTest(const thirdai::hashing::HashFunction& hash,
                            Similarity& sim, uint32_t dim, uint32_t num_tables,
                            uint32_t num_tests, float max_diff,
                            float max_avg_diff);

/**
 * This test hashes num_tests pairs of dense vectors at uniformly distributed
 * similarities using the HashFunctions dense and sparse hashing functions,
 * and ensures that the hashes are equal.
 */
void runSparseDenseEqTest(const thirdai::hashing::HashFunction& hash,
                          Similarity& sim, uint32_t dim, uint32_t num_tables,
                          uint32_t num_tests);

}  // namespace thirdai::hashing
