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
class SegmentedSparseFeatureVector final : public SegmentedFeatureVector {
 public:
  explicit SegmentedSparseFeatureVector(bool store_segment_feature_map = false)
      : SegmentedFeatureVector(store_segment_feature_map) {}

  bool empty() const final { return _indices.empty(); }

  void addSparseFeatureToSegment(uint32_t index, float value) final {
    if (_n_dense_added > 0) {
      throw std::invalid_argument(
          "[SegmentedSparseFeatureVector::addSparseFeatureToSegment] A block "
          "cannot "
          "add both dense and sparse features.");
    }

    uint32_t concat_index = _current_starting_dim + index;
    if (concat_index >= _current_ending_dim) {
      std::stringstream ss;
      ss << "[SegmentedSparseFeatureVector::addSparseFeatureToSegment] Setting "
            "value at index = "
         << index << " of vector segment with dim = "
         << _current_ending_dim - _current_starting_dim;
      throw std::invalid_argument(ss.str());
    }

    // We don't check whether we've seen this index before.
    // This is fine because bolt iterates through all index-value pairs of
    // sparse input vectors, so duplicates are effectively summed.
    _indices.push_back(concat_index);
    _values.push_back(value);
    _added_sparse = true;

    if (_store_index_to_segment_feature_map) {
      _index_to_segment_feature.emplace(
          concat_index, SegmentFeature(
                            /* segment_idx= */ _n_segments_added - 1,
                            /* feature_idx= */ index));
    }
  }

  void addDenseFeatureToSegment(float value) final {
    if (_added_sparse) {
      throw std::invalid_argument(
          "[SegmentedSparseFeatureVector::addDenseFeatureToSegment] A block "
          "cannot "
          "add both dense and sparse features.");
    }

    if (_n_dense_added >= (_current_ending_dim - _current_starting_dim)) {
      std::stringstream ss;
      ss << "[SegmentedSparseFeatureVector::addDenseFeatureToSegment] Adding "
         << _n_dense_added + 1
         << "-th dense feature to vector segment with dim = "
         << _current_ending_dim - _current_starting_dim;
      throw std::invalid_argument(ss.str());
    }

    uint32_t orig_index = _n_dense_added;
    uint32_t concat_index = _current_starting_dim + orig_index;

    _indices.push_back(concat_index);
    _values.push_back(value);
    _n_dense_added++;

    if (_store_index_to_segment_feature_map) {
      _index_to_segment_feature.emplace(
          concat_index, SegmentFeature(
                            /* segment_idx= */ _n_segments_added - 1,
                            /* feature_idx= */ orig_index));
    }
  }

  BoltVector toBoltVector() final {
    return BoltVector::makeSparseVector(_indices, _values);
  }

  IndexToSegmentFeatureMap getIndexToSegmentFeatureMapImpl() final {
    return _index_to_segment_feature;
  }

  void addFeatureSegment(uint32_t dim) final {
    _n_segments_added++;
    _current_starting_dim = _current_ending_dim;
    _current_ending_dim += dim;
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
  uint32_t _n_segments_added = 0;
  uint32_t _n_dense_added = 0;
  uint32_t _current_ending_dim = 0;
  uint32_t _current_starting_dim = 0;
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
};

/**
 * A concrete implementation of SegmentedFeatureVector where features
 * are not concatenated but instead hashed to the same range with a
 * different salt for each segment.
 */
class HashedSegmentedFeatureVector final : public SegmentedFeatureVector {
 public:
  explicit HashedSegmentedFeatureVector(uint32_t hash_range,
                                        bool store_segment_feature_map = false)
      : SegmentedFeatureVector(store_segment_feature_map),
        _hash_range(hash_range) {}

  bool empty() const final { return _indices.empty(); }

  void addSparseFeatureToSegment(uint32_t index, float value) final {
    if (_n_dense_added > 0) {
      throw std::invalid_argument(
          "[HashedSegmentedFeatureVector::addSparseFeatureToSegment] A block "
          "cannot add both dense and sparse features.");
    }

    // We don't check whether we've seen this index before.
    // This is fine because bolt iterates through all index-value pairs of
    // sparse input vectors, so duplicates are effectively summed.

    auto hashed_index = getHashedIndex(index);

    _indices.push_back(hashed_index);
    _values.push_back(value);
    _added_sparse = true;

    if (_store_index_to_segment_feature_map) {
      _index_to_segment_feature.emplace(
          hashed_index, SegmentFeature(
                            /* segment_idx= */ _n_segments_added - 1,
                            /* feature_idx= */ index));
    }
  }

  void addDenseFeatureToSegment(float value) final {
    if (_added_sparse) {
      throw std::invalid_argument(
          "[HashedSegmentedFeatureVector::addDenseFeatureToSegment] A block "
          "cannot add both dense and sparse features.");
    }

    uint32_t index = _n_dense_added;
    uint32_t hashed_index = getHashedIndex(_n_dense_added);

    _indices.push_back(hashed_index);
    _values.push_back(value);
    _n_dense_added++;

    if (_store_index_to_segment_feature_map) {
      _index_to_segment_feature.emplace(
          hashed_index, SegmentFeature(
                            /* segment_idx= */ _n_segments_added - 1,
                            /* feature_idx= */ index));
    }
  }

  BoltVector toBoltVector() final {
    return BoltVector::makeSparseVector(_indices, _values);
  }

  IndexToSegmentFeatureMap getIndexToSegmentFeatureMapImpl() final {
    return _index_to_segment_feature;
  }

  void addFeatureSegment(uint32_t dim) final {
    (void)dim;
    _added_sparse = false;
    _n_dense_added = 0;
    _n_segments_added++;
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
    return hashing::combineHashes(index, _n_segments_added) % _hash_range;
  }

  uint32_t _hash_range;

  bool _added_sparse = false;
  uint32_t _n_dense_added = 0;
  uint32_t _n_segments_added = 0;

  std::vector<uint32_t> _indices;
  std::vector<float> _values;
};

/**
 * A concrete implementation of SegmentedFeatureVector for dense vectors.
 */
class SegmentedDenseFeatureVector final : public SegmentedFeatureVector {
 public:
  explicit SegmentedDenseFeatureVector(bool store_segment_feature_map = false)
      : SegmentedFeatureVector(store_segment_feature_map) {}

  bool empty() const final { return _values.empty(); }

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

    uint32_t index = _n_dense_added;
    uint32_t concat_index = _values.size();
    _values.push_back(value);
    _n_dense_added++;

    if (_store_index_to_segment_feature_map) {
      _index_to_segment_feature.emplace(
          concat_index, SegmentFeature(
                            /* segment_idx= */ _n_segments_added - 1,
                            /* feature_idx= */ index));
    }
  }

  BoltVector toBoltVector() final {
    return BoltVector::makeDenseVector(_values);
  };

  IndexToSegmentFeatureMap getIndexToSegmentFeatureMapImpl() final {
    return _index_to_segment_feature;
  }

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

    _n_segments_added++;
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
  uint32_t _n_segments_added = 0;
  uint32_t _latest_segment_dim = 0;
  uint32_t _n_dense_added = 0;
  std::vector<float> _values;
};

}  // namespace thirdai::dataset