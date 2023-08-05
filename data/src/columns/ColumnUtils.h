#pragma once

#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
inline std::vector<T> permuteVector(const std::vector<T>& vector,
                                    const std::vector<size_t>& permutation) {
  std::exception_ptr permutation_err;
  std::vector<T> new_vector(permutation.size());
#pragma omp parallel for default(none) shared(vector, permutation)
  for (size_t i = 0; i < new_vector.size(); ++i) {
    if (permutation[i] >= vector.size()) {
      std::stringstream error_ss;
      error_ss << "Invalid permutation. Original vector has " << vector.size()
               << " elements but permutation contains index " << permutation[i]
               << ".";
#pragma omp critical
      permutation_err =
          std::make_exception_ptr(std::invalid_argument(error_ss.str()));
      continue;
    }

    new_vector[i] = vector[permutation[i]];
  }
  std::rethrow_exception(permutation_err);
}  // namespace thirdai::data

template <typename T>
inline std::vector<T> concatVectors(std::vector<T>&& a, std::vector<T>&& b) {
  if (&a == &b) {
    throw std::invalid_argument("Cannot concatenate a column with itself.");
  }
  std::vector<T> new_vec(a.size() + b.size());

  for (size_t i = 0; i < a.size(); i++) {
    new_vec[i] = std::move(a[i]);
  }

  for (size_t i = 0; i < b.size(); i++) {
    new_vec[a.size() + i] = std::move(b[i]);
  }

  return new_vec;
}

template <typename T>
inline std::pair<std::vector<T>, std::vector<T>> splitVector(
    std::vector<T>&& vector, size_t starting_offset) {
  if (starting_offset >= vector.size()) {
    throw std::invalid_argument(
        "invalid split offset " + std::to_string(starting_offset) +
        " for column of length " + std::to_string(vector.size()) + ".");
  }

  std::vector<T> front(
      std::make_move_iterator(vector.begin()),
      std::make_move_iterator(vector.begin() + starting_offset));
  std::vector<T> back(std::make_move_iterator(vector.begin()) + starting_offset,
                      std::make_move_iterator(vector.end()));

  return std::make_pair(std::move(front), std::move(back));
}

}  // namespace thirdai::data