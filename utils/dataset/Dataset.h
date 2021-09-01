#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::utils {

enum class BATCH_TYPE { SPARSE, SPARSE_LABELED, DENSE, DENSE_LABELED };

struct Batch {
 public:
  // TODO: Comment these methods
  uint32_t _batch_size{0};
  uint32_t** _indices{nullptr};
  float** _values{nullptr};
  uint32_t* _lens{nullptr};
  uint32_t** _labels{nullptr};
  uint32_t* _label_lens{nullptr};
  BATCH_TYPE _type;
  uint32_t _dim;

  /** Creates a new Batch object with a size, data dimension, and data type */
  Batch(uint64_t batch_size, BATCH_TYPE type, uint32_t dim) {
    assert(_dim != 0);

    _type = type;
    _batch_size = batch_size;
    _values = new float*[_batch_size];
    _dim = dim;

    if (_type == BATCH_TYPE::SPARSE || _type == BATCH_TYPE::SPARSE_LABELED) {
      _indices = new uint32_t*[_batch_size];
      _lens = new uint32_t[_batch_size];
    }
    if (_type == BATCH_TYPE::SPARSE_LABELED ||
        _type == BATCH_TYPE::DENSE_LABELED) {
      _labels = new uint32_t*[_batch_size];
      _label_lens = new uint32_t[_batch_size];
    }
  }

  // No copy constructor
  Batch(const Batch& other) = delete;

  // Move constructor
  Batch(Batch&& other) noexcept
      : _batch_size(other._batch_size),
        _indices(other._indices),
        _values(other._values),
        _lens(other._lens),
        _labels(other._labels),
        _label_lens(other._label_lens),
        _type(other._type),
        _dim(other._dim) {
    // Set fields to null so we do not delete the fields of our current object
    other._batch_size = 0;
    other._indices = nullptr;
    other._values = nullptr;
    other._lens = nullptr;
    other._labels = nullptr;
    other._label_lens = nullptr;
  }

  // No copy assignment operator
  Batch& operator=(const Batch& other) = delete;

  // Move assignment operator
  Batch& operator=(Batch&& other) noexcept {
    _batch_size = other._batch_size;
    _indices = other._indices;
    _values = other._values;
    _lens = other._lens;
    _labels = other._labels;
    _label_lens = other._label_lens;

    other._batch_size = 0;
    other._indices = nullptr;
    other._values = nullptr;
    other._lens = nullptr;
    other._labels = nullptr;
    other._label_lens = nullptr;
    return *this;
  }

  ~Batch() {
    delete[] _indices;
    delete[] _lens;
    delete[] _values;
    delete[] _labels;
    delete[] _label_lens;
  }
};

class Dataset {
 public:
  /**
   * Returns nullptr if there are no more batches, and otherwise
   * a pointer to the next batch). The caller is responsible for checking
   * if the Batch is a nullptr and also freeing the Batch memory when it no longer
   * needs it.
   */
  virtual Batch *getNextBatch() = 0;
};

}  // namespace thirdai::utils