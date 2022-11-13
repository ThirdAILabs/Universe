#pragma once

#include <hashing/src/HashUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::dataset {
/**
 * A concrete implementation of SegmentedSparseFeatureVector for sparse vectors.
 */
class SegmentedSparseFeatureVector : public SegmentedFeatureVector {
 public:
  void addSparseFeatureToSegment(uint32_t index, float value) final {
    if (_n_dense_added > 0) {
      throw std::invalid_argument(
          "[SegmentedSparseFeatureVector::addSparseFeatureToSegment] A block "
          "cannot "
          "add both dense and sparse features.");
    }

    uint32_t concat_index = _prev_dim + index;
    if (concat_index >= _current_dim) {
      std::stringstream ss;
      ss << "[SegmentedSparseFeatureVector::addSparseFeatureToSegment] Setting "
            "value "
            "at index = "
         << index
         << " of vector segment with dim = " << _current_dim - _prev_dim;
      throw std::invalid_argument(ss.str());
    }

    // We don't check whether we've seen this index before.
    // This is fine because bolt iterates through all index-value pairs of
    // sparse input vectors, so duplicates are effectively summed.
    _indices.push_back(concat_index);
    _values.push_back(value);
    _added_sparse = true;
  }

  void addDenseFeatureToSegment(float value) final {
    if (_added_sparse) {
      throw std::invalid_argument(
          "[SegmentedSparseFeatureVector::addDenseFeatureToSegment] A block "
          "cannot "
          "add both dense and sparse features.");
    }

    if (_n_dense_added >= (_current_dim - _prev_dim)) {
      std::stringstream ss;
      ss << "[SegmentedSparseFeatureVector::addDenseFeatureToSegment] Adding "
         << _n_dense_added + 1
         << "-th dense feature to vector segment with dim = "
         << _current_dim - _prev_dim;
      throw std::invalid_argument(ss.str());
    }

    _indices.push_back(_prev_dim + _n_dense_added);
    _values.push_back(value);
    _n_dense_added++;
  }

  BoltVector toBoltVector() final {
    return BoltVector::makeSparseVector(_indices, _values);
  }

  void addFeatureSegment(uint32_t dim) final {
    _prev_dim = _current_dim;
    _current_dim += dim;
    _added_sparse = false;
    _n_dense_added = 0;
  }

 protected:
  std::unordered_map<uint32_t, float> entries() final {
    std::unordered_map<uint32_t, float> ents;
    for (uint32_t i = 0; i < _indices.size(); i++) {
      ents[_indices[i]] += _values[i];
    }
    return ents;
  }

 private:
  bool _added_sparse = false;
  uint32_t _n_dense_added = 0;
  uint32_t _current_dim = 0;
  uint32_t _prev_dim = 0;
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
};

/**
 * A concrete implementation of SegmentedFeatureVector where features 
 * are not concatenated but instead hashed to the same range with a
 * different salt for each segment.
 */
class HashedSegmentedFeatureVector : public SegmentedFeatureVector {
 public:
  explicit HashedSegmentedFeatureVector(uint32_t hash_range)
      : _hash_range(hash_range) {}

  void addSparseFeatureToSegment(uint32_t index, float value) final {
    if (_n_dense_added > 0) {
      throw std::invalid_argument(
          "[HashedSegmentedFeatureVector::addSparseFeatureToSegment] A block "
          "cannot add both dense and sparse features.");
    }

    // We don't check whether we've seen this index before.
    // This is fine because bolt iterates through all index-value pairs of
    // sparse input vectors, so duplicates are effectively summed.
    _indices.push_back(getHashedIndex(index));
    _values.push_back(value);
    _added_sparse = true;
  }

  void addDenseFeatureToSegment(float value) final {
    if (_added_sparse) {
      throw std::invalid_argument(
          "[HashedSegmentedFeatureVector::addDenseFeatureToSegment] A block "
          "cannot add both dense and sparse features.");
    }

    _indices.push_back(getHashedIndex(_n_dense_added));
    _values.push_back(value);
    _n_dense_added++;
  }

  BoltVector toBoltVector() final {
    return BoltVector::makeSparseVector(_indices, _values);
  }

  void addFeatureSegment(uint32_t dim) final {
    (void)dim;
    _added_sparse = false;
    _n_dense_added = 0;
    _block_count++;
  }

 protected:
  std::unordered_map<uint32_t, float> entries() final {
    std::unordered_map<uint32_t, float> ents;
    for (uint32_t i = 0; i < _indices.size(); i++) {
      ents[_indices[i]] += _values[i];
    }
    return ents;
  }

 private:
  uint32_t getHashedIndex(uint32_t index) const {
    return hashing::HashUtils::combineHashes(index, _block_count) % _hash_range;
  }

  uint32_t _hash_range;

  bool _added_sparse = false;
  uint32_t _n_dense_added = 0;
  uint32_t _block_count = 0;

  std::vector<uint32_t> _indices;
  std::vector<float> _values;
};

/**
 * A concrete implementation of SegmentedFeatureVector for dense vectors.
 */
class SegmentedDenseFeatureVector : public SegmentedFeatureVector {
 public:
  void addSparseFeatureToSegment(uint32_t index, float value) final {
    (void)index;
    (void)value;
    throw std::invalid_argument(
        "[SegmentedDenseFeatureVector::addSparseFeatureToSegment] "
        "SegmentedDenseFeatureVector does not accept sparse features.");
  }

  void addDenseFeatureToSegment(float value) final {
    if (_n_dense_added >= _latest_segment_dim) {
      std::stringstream ss;
      ss << "[SegmentedDenseFeatureVector::addDenseFeatureToSegment] Adding "
         << _n_dense_added + 1
         << "-th dense feature to vector segment with dim = "
         << _latest_segment_dim;
      throw std::invalid_argument(ss.str());
    }

    _values.push_back(value);
    _n_dense_added++;
  }

  BoltVector toBoltVector() final {
    return BoltVector::makeDenseVector(_values);
  };

  void addFeatureSegment(uint32_t dim) final {
    if (_latest_segment_dim > _n_dense_added) {
      std::stringstream ss;
      ss << "[SegmentedDenseFeatureVector::addFeatureSegment] Adding vector "
            "segment before "
            "completing previous segment. Previous segment expected to "
            "have dim = "
         << _latest_segment_dim << " but only " << _n_dense_added
         << " dense features were added.";
      throw std::invalid_argument(ss.str());
    }

    _latest_segment_dim = dim;
    _n_dense_added = 0;
    _values.reserve(_values.size() + dim);
  }

 protected:
  std::unordered_map<uint32_t, float> entries() final {
    std::unordered_map<uint32_t, float> ents;
    for (uint32_t i = 0; i < _values.size(); i++) {
      ents[i] += _values[i];
    }
    return ents;
  }

 private:
  uint32_t _latest_segment_dim = 0;
  uint32_t _n_dense_added = 0;
  std::vector<float> _values;
};

}  // namespace thirdai::dataset