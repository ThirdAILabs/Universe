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

    // We don't check whether we've seen this index before.
    // This is fine because bolt iterates through all index-value pairs of
    // sparse input vectors, so duplicates are effectively summed.
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
    (void)index;
    (void)value;
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