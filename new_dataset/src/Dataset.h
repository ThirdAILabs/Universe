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