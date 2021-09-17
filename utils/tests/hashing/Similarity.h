#pragma once

#include "DenseVector.h"
#include "SparseVector.h"
#include <cassert>
#include <iostream>
#include <math.h>
#include <random>
#include <unordered_set>
#include <utility>

// TODO(josh): Add DWTA and Jaccard similarities and test those hash classes
// TOOD(josh): Add Euclidean similarity
namespace thirdai::utils::lsh_testing {

struct SparseVecPair {
  SparseVector v1;
  SparseVector v2;
  float sim;
};

struct DenseVecPair {
  DenseVector v1;
  DenseVector v2;
  float sim;
};

/**
 * Represents a similarity over a metric space. Similarity should be a measure
 * from 0 to 1 and should be inversely related to distance. Implementations
 * should have an internal random device as part of their state so that calls
 * to the methods do not return the same vectors every time.
 */
class Similarity {
 public:
  /**
   * This method generates two random dense vectors with dimension dim that
   * have approximate similarity equal to sim. Some similarities (e.g. cosine)
   * will allow the generation of two vectors with the exact similarity input,
   * but for some similarities this is much more difficult (e.g. Jaccard) and
   * the best we can hope for is an approximate. Thus, we return the two
   * vectors, along with the actual similarity between the two vectors.
   */
  virtual DenseVecPair getRandomDenseVectors(float sim, uint32_t dim) = 0;

  /**
   * This method generates two random sparse vectors with overall max dimension
   * dim and num_non_zeros total number of non zeros. Otherwise, this method
   * is the exact same as getRandomDenseVectors.
   */
  virtual SparseVecPair getRandomSparseVectors(float sim,
                                               uint32_t num_non_zeros,
                                               uint32_t dim) = 0;

  /** Returns the similarity of the two dense input vectors. */
  virtual float getSim(const DenseVector& v1, DenseVector& v2) = 0;

  /** Returns the similarity of the two sparse input vectors. */
  virtual float getSim(const SparseVector& v1, const SparseVector& v2) = 0;
};

}  // namespace thirdai::utils::lsh_testing
