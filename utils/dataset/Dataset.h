#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace thirdai::utils {

enum class BATCH_TYPE { SPARSE, DENSE };
enum class LABEL_TYPE { LABELED, UNLABELED };

struct Batch {
 public:
  // TODO(any): Comment these methods
  uint32_t _batch_size{0};
  uint32_t** _indices{nullptr};
  float** _values{nullptr};
  uint32_t* _lens{nullptr};
  uint32_t** _labels{nullptr};
  uint32_t* _label_lens{nullptr};
  BATCH_TYPE _batch_type;
  LABEL_TYPE _label_type;
  uint32_t _dim;

  /** Default constructor */
  Batch(){};

  /** Creates a new Batch object with a size, data dimension, and data type */
  Batch(uint64_t batch_size, BATCH_TYPE batch_type, LABEL_TYPE label_type,
        uint32_t dim) {
    assert(_dim != 0);

    _batch_type = batch_type;
    _label_type = label_type;
    _batch_size = batch_size;
    _values = new float*[_batch_size];
    _dim = dim;

    if (_batch_type == BATCH_TYPE::SPARSE) {
      _indices = new uint32_t*[_batch_size];
      _lens = new uint32_t[_batch_size];
    }
    if (_label_type == LABEL_TYPE::LABELED) {
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
        _batch_type(other._batch_type),
        _label_type(other._label_type),
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
   * target_batch_num_per_read is the number of batches loaded from file each
   * time. set to 0 to load entire file.
   */
  Dataset(uint64_t target_batch_size, uint64_t target_batch_num_per_load)
      : _target_batch_size(target_batch_size),
        _target_batch_num_per_load(target_batch_num_per_load){};

  /**
   * The batch is only going to exist until the next call to loadNextBatchSet();
   */
  const Batch& operator[](uint64_t i) const {
    assert(i <= _num_batches);
    return _batches[i];
  }

  /**
   * Load n batches from file
   * Frees any memory for currently existing batches and updates _numBatches.
   */
  virtual void loadNextBatchSet() = 0;

  /**
   * Number of batches currently loaded
   */
  uint64_t numBatches() const { return _num_batches; };

  ~Dataset() { delete[] _batches; }

 protected:
  const uint64_t _target_batch_size, _target_batch_num_per_load;
  uint64_t _num_batches;
  Batch* _batches;
};

}  // namespace thirdai::utils