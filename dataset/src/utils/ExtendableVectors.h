#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <sstream>
#include <stdexcept>

namespace thirdai::dataset {
/**
 * A concrete implementation of ExtendableVector for sparse vectors.
 */
struct SparseExtendableVector : public ExtendableVector {

  void addExtensionSparseFeature(uint32_t index, float value) final {
    if (_n_dense_added > 0) {
      throw std::invalid_argument(
          "[SparseExtendableVector::addExtensionSparseFeature] A block cannot "
          "add both dense and sparse features.");
    }

    uint32_t concat_index = _prev_dim + index;
    if (concat_index >= _current_dim) {
      std::stringstream ss;
      ss << "[SparseExtendableVector::addExtensionSparseFeature] Setting value "
            "at index = "
         << index
         << " of extension vector with dim = " << _current_dim - _prev_dim;
      throw std::invalid_argument(ss.str());
    }

    _indices.push_back(concat_index);
    _values.push_back(value);
    _added_sparse = true;
  }

  void addExtensionDenseFeature(float value) final {
    if (_added_sparse) {
      throw std::invalid_argument(
          "[SparseExtendableVector::addExtensionDenseFeature] A block cannot "
          "add both dense and sparse features.");
    }

    if (_n_dense_added >= (_current_dim - _prev_dim)) {
      std::stringstream ss;
      ss << "[SparseExtendableVector::addExtensionDenseFeature] Adding "
         << _n_dense_added + 1
         << "-th dense feature to extension vector with dim = "
         << _current_dim - _prev_dim;
      throw std::invalid_argument(ss.str());
    }

    _indices.push_back(_prev_dim + _n_dense_added);
    _values.push_back(value);
    _n_dense_added++;
  }

  /**
   * Deduplicates indices by summing values.
   */
  void incrementExtensionAtIndices(std::vector<uint32_t>& indices,
                                   float inc) final {
    // Put equivalent indices next to each other.
    std::sort(indices.begin(), indices.end());

    /**
     * If current index is the same as the next index, keep accumulating
     * val. Otherwise, add sparse feature at the current index with the
     * accumulated value and reset val.
     */
    float val = 0.0;
    uint32_t i = 0;
    for (; i < indices.size() - 1; ++i) {
      uint32_t idx = indices[i];
      uint32_t next_idx = indices[i + 1];
      val += inc;

      if (idx != next_idx) {
        addExtensionSparseFeature(idx, val);
        val = 0.0;  // Reset val since next idx is different.
      }
    }

    /**
     * If we're looking at the last element, the next element is clearly
     * "different", so we add a sparse feature accordingly.
     */
    if (i == indices.size() - 1) {
      val += inc;
      addExtensionSparseFeature(indices.back(), val);
    }
  }

  bolt::BoltVector toBoltVector() final {
    return bolt::BoltVector::makeSparseVector(_indices, _values);
  }

 protected:
  void extendByDim(uint32_t dim) final {
    _prev_dim = _current_dim;
    _current_dim += dim;
    _added_sparse = false;
    _n_dense_added = 0;
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
 * A concrete implementation of ExtendableVector for dense vectors.
 */
struct DenseExtendableVector : public ExtendableVector {

  void addExtensionSparseFeature(uint32_t index, float value) final {
    throw std::invalid_argument(
        "[DenseExtendableVector::addExtensionSparseFeature] "
        "DenseExtendableVector does not accept sparse features.");
  }

  void addExtensionDenseFeature(float value) final {
    if (_n_dense_added >= _latest_extension_dim) {
      std::stringstream ss;
      ss << "[SparseExtendableVector::addExtensionDenseFeature] Adding "
         << _n_dense_added + 1
         << "-th dense feature to extension vector with dim = "
         << _latest_extension_dim;
      throw std::invalid_argument(ss.str());
    }

    _values.push_back(value);
    _n_dense_added++;
  }

  void incrementExtensionAtIndices(std::vector<uint32_t>& indices,
                                   float inc) final {
    (void)indices;
    (void)inc;

    throw std::invalid_argument(
        "[DenseExtendableVector::incrementAtIndices] DenseExtendableVector "
        "does not "
        "support this operation.");
  }

  bolt::BoltVector toBoltVector() final {
    return bolt::BoltVector::makeDenseVector(_values);
  };

 protected:
  void extendByDim(uint32_t dim) final {
    if (_latest_extension_dim > _n_dense_added) {
      std::stringstream ss;
      ss << "[SparseExtendableVector::extendByDim] Extending vector before "
            "completing previous extension. Previous extension expected to "
            "have dim = "
         << _latest_extension_dim << " but only " << _n_dense_added
         << " dense features were added.";
      throw std::invalid_argument(ss.str());
    }

    _latest_extension_dim = dim;
    _n_dense_added = 0;
    _values.resize(_values.size() + dim);
  }

 private:
  uint32_t _latest_extension_dim = 0;
  uint32_t _n_dense_added = 0;
  std::vector<float> _values;
};

}  // namespace thirdai::dataset