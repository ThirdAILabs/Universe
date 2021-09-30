#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::utils {

enum class BATCH_TYPE { SPARSE, DENSE };
enum class LABEL_TYPE { LABELED, UNLABELED };
enum class ID_TYPE { SEQUENTIAL, INDIVIDUAL };

struct Batch {
 public:
  // TODO(any): Comment these methods
  uint32_t _batch_size{0};
  uint32_t** _indices{nullptr};
  float** _values{nullptr};
  uint32_t* _lens{nullptr};
  uint32_t** _labels{nullptr};
  uint32_t* _label_lens{nullptr};
  uint64_t _starting_id{0};
  uint64_t* _individual_ids{nullptr};
  BATCH_TYPE _batch_type{BATCH_TYPE::SPARSE};
  LABEL_TYPE _label_type{LABEL_TYPE::LABELED};
  ID_TYPE _id_type{ID_TYPE::INDIVIDUAL};
  uint32_t _dim{0};

  /** Default constructor */
  Batch(){};

  /**
   * Creates a new Batch object with a size, data dimension, and data type.
   * If sparse, dimension can be set to 0. The _indices, _values, _lens,
   * _labels, _label_lens, and either _starting_id or _individual_ids (depending
   * on the ID_TYPE passed in) will need to be set after construction, although
   * _indices, _values, _lens, _labels, _label_lens, and _individual_ids
   * will be initialized to empty arrays of the correct size.
   */
  Batch(uint64_t batch_size, BATCH_TYPE batch_type, LABEL_TYPE label_type,
        ID_TYPE id_type, uint32_t dim) {
    // TODO(any): For some reason putting these in an initializer list
    // causes a segfault, this is probably bad memory management somewhere.
    _batch_size = batch_size;
    _batch_type = batch_type;
    _label_type = label_type;
    _id_type = id_type;

    if (_batch_type == BATCH_TYPE::DENSE && dim == 0) {
      throw std::invalid_argument("Dense batch does not accept dim = 0");
    }

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
    if (_id_type == ID_TYPE::INDIVIDUAL) {
      _individual_ids = new uint64_t[_batch_size];
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
        _starting_id(other._starting_id),
        _individual_ids(other._individual_ids),
        _batch_type(other._batch_type),
        _label_type(other._label_type),
        _id_type(other._id_type),
        _dim(other._dim) {
    // Set fields to null so we do not delete the fields of our current object
    other._indices = nullptr;
    other._values = nullptr;
    other._lens = nullptr;
    other._labels = nullptr;
    other._label_lens = nullptr;
    other._individual_ids = nullptr;
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
    _starting_id = other._starting_id;
    _individual_ids = other._individual_ids;
    _batch_type = other._batch_type;
    _label_type = other._label_type;
    _id_type = other._id_type;
    _dim = other._dim;

    other._indices = nullptr;
    other._values = nullptr;
    other._lens = nullptr;
    other._labels = nullptr;
    other._label_lens = nullptr;
    other._individual_ids = nullptr;
    return *this;
  }

  ~Batch() {
    delete[] _indices;
    delete[] _lens;
    delete[] _values;
    delete[] _labels;
    delete[] _label_lens;
    delete[] _individual_ids;
  }
};

class Dataset {
 public:
  /**
   * target_batch_num_per_read is the number of batches loaded from file each
   * time. set to 0 to load entire file.
   * Calling the constructor should not load the first batch set.
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

  virtual ~Dataset() { delete[] _batches; }

 protected:
  const uint64_t _target_batch_size, _target_batch_num_per_load;
  uint64_t _num_batches;
  // In the future, we may need two batch arrays if we want to read in parallel
  // while processing
  Batch* _batches{nullptr};
};

}  // namespace thirdai::utils