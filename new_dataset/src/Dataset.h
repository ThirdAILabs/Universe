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

// TODO(Josh): Describe design
// The only downside of this design is that while any dataset slice pointing
// to the original entire dataset exists, the entire dataset will stay in
// memory.
class Dataset {
 public:
  explicit Dataset(UnderlyingDataset&& vectors)
      : _vectors(std::make_shared<UnderlyingDataset>(vectors)),
        _start_index(0),
        _length(_vectors->size()) {}

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
    _vectors->at(_start_index + i) = vector;
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
    return _vectors->end() + _start_index + _length;
  }

 private:
  explicit Dataset(UnderlyingDatasetPtr vectors, uint64_t start_index,
                   uint64_t length)
      : _vectors(std::move(vectors)),
        _start_index(start_index),
        _length(length) {}

  inline BoltVector& getWithBoundsCheck(uint64_t i) const {
    checkBounds(i);
    return (*_vectors)[_start_index + i];
  }

  inline void checkBounds(uint64_t i) const {
    if (i >= _length) {
      throw std::invalid_argument("Requesting vector with index " +
                                  std::to_string(i) + ", but there are only " +
                                  std::to_string(_length) +
                                  " vectors in this Dataset.");
    }
  }

  UnderlyingDatasetPtr _vectors;
  uint64_t _start_index, _length;
};

}  // namespace thirdai::dataset