#pragma once

#include "DenseVectorUtils.h"
#include "Similarity.h"
#include "SparseVector.h"
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace thirdai::hashing {

class JaccardSim : public Similarity {
 public:
  explicit JaccardSim(uint32_t seed) : _generator(seed) {}

  DenseVecPair getRandomDenseVectors(float sim, uint32_t dim) override {
    (void)sim;
    (void)dim;
    throw thirdai::exceptions::NotImplemented(
        "Jaccard similarity is only for sparse vectors, cannot call "
        "getRandomDenseVectors on JaccardSim");
  }

  SparseVecPair getRandomSparseVectors(float sim, uint32_t num_non_zeros,
                                       uint32_t dim) override {
    // If we let s = sim, n = num_non_zeros, and k = size of the intersection,
    // we want s = k / (2n - k). Solving for k, we have that the optimal size
    // of the intersection is 2ns / (1 + s).
    uint32_t intersection_size = 2 * num_non_zeros * sim / (1 + sim);
    uint32_t union_size = 2 * num_non_zeros - intersection_size;

    // Since we will do rejection sampling we want the dimension to be
    // at least 2 times bigger than necessary.
    if (2 * union_size > dim) {
      throw std::invalid_argument(
          "Dimension " + std::to_string(dim) +
          " is not big enough to represent jaccard sim " + std::to_string(sim));
    }

    std::unordered_set<uint32_t> random_values_set;
    while (random_values_set.size() < union_size) {
      random_values_set.insert(_generator() % dim);
    }
    std::vector<uint32_t> random_values(random_values_set.begin(),
                                        random_values_set.end());

    std::vector<uint32_t> indices_1;
    std::vector<uint32_t> indices_2;
    for (uint32_t same_index = 0; same_index < intersection_size;
         same_index++) {
      uint32_t same_value = random_values.back();
      random_values.pop_back();
      indices_1.push_back(same_value);
      indices_2.push_back(same_value);
    }
    for (uint32_t diff_index = 0;
         diff_index < num_non_zeros - intersection_size; diff_index++) {
      uint32_t value_1 = random_values.back();
      random_values.pop_back();
      indices_1.push_back(value_1);
      uint32_t value_2 = random_values.back();
      random_values.pop_back();
      indices_2.push_back(value_2);
    }

    std::sort(indices_1.begin(), indices_1.end());
    std::sort(indices_2.begin(), indices_2.end());

    std::vector<float> empty_values;
    SparseVector v1(
        num_non_zeros);  // = {indices_1, empty_values, num_non_zeros};
    SparseVector v2(
        num_non_zeros);  // = {indices_2, empty_values, num_non_zeros};
    std::copy(indices_1.begin(), indices_1.end(), v1.indices.begin());
    std::copy(indices_2.begin(), indices_2.end(), v2.indices.begin());

    return {v1, v2,
            getJaccardSim(v1.indices.data(), v1.length(), v2.indices.data(),
                          v2.length())};
  }

  float getSim(const DenseVector& v1, DenseVector& v2) override {
    (void)v1;
    (void)v2;
    throw thirdai::exceptions::NotImplemented(
        "Jaccard similarity is only for sparse vectors, cannot call getSim on "
        "JaccardSim");
  }

  float getSim(const SparseVector& v1, const SparseVector& v2) override {
    return getJaccardSim(v1.indices.data(), v1.length(), v2.indices.data(),
                         v2.length());
  }

  /** Returns the jacard similarity of two sets, represented as sorted vectors
   */
  static float getJaccardSim(const uint32_t* s1, uint32_t l1,
                             const uint32_t* s2, uint32_t l2) {
    uint32_t intersection_size = 0;
    uint32_t s1_index = 0;
    for (uint32_t j = 0; j < l2; j++) {
      uint32_t s2_value = s2[j];
      while (s1_index < l1 && s1[s1_index] < s2_value) {
        s1_index++;
      }
      if (s1_index < l1 && s1[s1_index] == s2_value) {
        intersection_size++;
      }
    }
    const uint32_t union_size = l1 + l2 - intersection_size;
    return intersection_size / static_cast<float>(union_size);
  }

 private:
  std::mt19937 _generator;
};

}  // namespace thirdai::hashing
