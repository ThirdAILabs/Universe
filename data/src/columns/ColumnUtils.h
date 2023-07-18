#pragma once

#include <vector>

namespace thirdai::data {

template <typename T>
inline std::vector<T> shuffleVector(std::vector<T>&& vector,
                                    const std::vector<size_t>& permutation) {
  if (permutation.size() != vector.size()) {
    throw std::invalid_argument(
        "Size of permutation must match the number of rows.");
  }

  std::vector<T> new_vector(vector.size());

#pragma omp parallel for default(none) shared(new_vector, vector, permutation)
  for (size_t i = 0; i < vector.size(); i++) {
    std::swap(new_vector[i], vector[permutation[i]]);
  }

  return new_vector;
}

template <typename T>
inline std::vector<T> concatVectors(std::vector<T>&& a, std::vector<T>&& b) {
  std::vector<T> new_vec(a.size() + b.size());

  for (size_t i = 0; i < a.size(); i++) {
    new_vec[i] = std::move(a[i]);
  }

  for (size_t i = 0; i < b.size(); i++) {
    new_vec[a.size() + i] = std::move(b[i]);
  }

  return new_vec;
}

}  // namespace thirdai::data