#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::dataset {

class Dataset;
using DatasetPtr = std::shared_ptr<Dataset>;

using UnderlyingDataset = std::vector<BoltVector>;
using UnderlyingDatasetPtr = std::shared_ptr<UnderlyingDataset>;

// TODO(Josh/Nick): This class is pretty similar to BoltBatch, so we should
// try to refactor them into the same class in the future.
/**
 * The Dataset class is at its heart a shared pointer to a vector of
 * BoltVectors, a starting index, and a length. This design allows there to be
 * multiple Dataset objects that all point to different slices of an underlying
 * vector of BoltVectors, and allows all Dataset objects to have a reference to
 * their data without needing to worry about memory management. One downside of
 * this design is that while any dataset slice pointing to the original entire
 * dataset exists, the entire dataset will stay in memory. Numpy does this too:
 * https://stackoverflow.com/questions/50195197/reduce-memory-usage-when-slicing-numpy-arrays.
 * For the use cases we envision this tradeoff seems acceptable. For instace,
 * during batch processing in training, each batch has a shorter
 * lifetime than the entire dataset.
 */
class Dataset {
 public:
  explicit Dataset(UnderlyingDataset&& vectors)
      : _vectors(std::make_shared<UnderlyingDataset>(std::move(vectors))),
        _start_index(0),
        _length(_vectors->size()) {}

  Dataset(UnderlyingDatasetPtr vectors, uint64_t start_index, uint64_t length)
      : _vectors(std::move(vectors)),
        _start_index(start_index),
        _length(length) {}

  /**
   * We chose not to implement std::vector at() style operations that check the
   * bounds and [] style operations that don't, instead having a single []
   * opeations to always check the bounds. We made this decision because the
   * type within the vector is always BoltVector, and doing anything
   * interesting with a BoltVector will ikely take longer than an if statement
   * If this is ever a bottelenck we can add std::vector semantics.
   */
  const BoltVector& operator[](uint64_t i) const {
    return getWithBoundsCheck(i);
  }

  BoltVector& operator[](uint64_t i) { return getWithBoundsCheck(i); }

  void set(uint64_t i, BoltVector&& vector) {
    checkBounds(i);
    (*_vectors)[_start_index + i] = vector;
  }

  DatasetPtr slice(uint64_t slice_start_index, uint64_t slice_end_index) const {
    if (slice_end_index <= slice_start_index) {
      throw std::invalid_argument(
          "Slices must have positive size, but found start index " +
          std::to_string(slice_start_index) + " and end index " +
          std::to_string(slice_end_index));
    }
    checkBounds(_start_index + slice_end_index - 1);

    return std::make_shared<Dataset>(
        _vectors,
        /* start_index = */ _start_index + slice_start_index,
        /* length = */ slice_end_index - slice_start_index);
  }

  uint64_t len() const { return _length; }

  UnderlyingDataset::iterator begin() const {
    return _vectors->begin() + _start_index;
  }

  UnderlyingDataset::iterator end() const {
    return _vectors->begin() + _start_index + _length;
  }

  /*
   * Throws an exception if the vector is not of the passed in
   * expected_dimension (for a sparse vector this just means none of the
   * active neurons are too large). "origin_string" should be a descriptive
   * string that tells the user where the error comes from if it is thrown, e.g.
   * something like "Passed in BoltVector too large for Input".
   */
  void verifyExpectedDimension(
      uint32_t expected_dimension,
      std::optional<std::pair<uint32_t, uint32_t>> num_nonzeros_range,
      const std::string& origin_string) const {
    for (BoltVector& vec : *this) {
      if (vec.isDense()) {
        if (vec.len != expected_dimension) {
          throw std::invalid_argument(
              origin_string + ": Received dense BoltVector with dimension=" +
              std::to_string(vec.len) +
              ", but was supposed to have dimension=" +
              std::to_string(expected_dimension));
        }
      } else {
        for (uint32_t i = 0; i < vec.len; i++) {
          uint32_t active_neuron = vec.active_neurons[i];
          if (active_neuron >= expected_dimension) {
            throw std::invalid_argument(
                origin_string +
                ": Received sparse BoltVector with active_neuron=" +
                std::to_string(active_neuron) + " but was supposed to have=" +
                std::to_string(expected_dimension));
          }
        }
      }

      if (num_nonzeros_range && (vec.len > num_nonzeros_range.value().second ||
                                 vec.len < num_nonzeros_range.value().first)) {
        std::stringstream ss;
        ss << origin_string << ": Received BoltVector with len "
           << std::to_string(vec.len) + " but was expected to have between "
           << num_nonzeros_range.value().first << " and "
           << num_nonzeros_range.value().second << " nonzeros.";

        throw std::invalid_argument(ss.str());
      }
    }
  }

 private:
  inline BoltVector& getWithBoundsCheck(uint64_t i) const {
    checkBounds(i);
    return (*_vectors)[_start_index + i];
  }

  inline void checkBounds(uint64_t i) const {
    if (i >= _length) {
      throw std::out_of_range("Requesting vector with index " +
                              std::to_string(i) + ", but there are only " +
                              std::to_string(_length) +
                              " vectors in this Dataset.");
    }
  }

  UnderlyingDatasetPtr _vectors;
  uint64_t _start_index, _length;
};

}  // namespace thirdai::dataset