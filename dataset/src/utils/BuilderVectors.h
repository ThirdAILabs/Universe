#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::dataset {

/**
 * Builder vector interface.
 * A builder vector is a data structure for composing features
 * from different blocks into a single vector.
 *
 */
struct BuilderVector {
  /**
   * Sets the start_dim-th dimension of the vector to value.
   */
  virtual void addSingleFeature(uint32_t start_dim, float value) = 0;

  /**
   * Sets the value at the given indices to the given values.
   */
  virtual void addSparseFeatures(std::vector<uint32_t>& indices,
                                 std::vector<float>& values) = 0;

  /**
   * Sets the value at the given indices to the given values.
   * This one takes numpy arrays.
   */
  virtual void addSparseFeaturesNumpy(
      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& indices,
      py::array_t<float, py::array::c_style | py::array::forcecast>&
          values) = 0;

  /**
   * Sets the value at the given indices to the given values.
   */
  virtual void addDenseFeatures(uint32_t start_dim,
                                std::vector<float>& values) = 0;

  /**
   * Sets the value at the given indices to the given values.
   * This one takes a numpy array.
   */
  virtual void addDenseFeaturesNumpy(
      uint32_t start_dim,
      py::array_t<float, py::array::c_style | py::array::forcecast>&
          values) = 0;

  /**
   * Given a sequence of possibly-repeating indices, increment values at these
   * indices by the given amount. Repetitions accumulate.
   */
  virtual void incrementAtIndices(std::vector<uint32_t>& indices,
                                  float inc) = 0;

  /**
   * Makes a BoltVector.
   */
  virtual bolt::BoltVector toBoltVector() = 0;

 protected:
  /**
   * Checks that the given numpy array is 1 dimensional.
   */
  static void checkNumpyArray1D(const py::buffer_info& buf) {
    if (buf.shape.size() != 1) {
      std::stringstream ss;
      ss << "[BuilderVector::checkNumpyArray1D] The given numpy array must be "
            "1 dimensional. Found "
         << buf.shape.size() << "-dimensional array.";
      throw std::invalid_argument(ss.str());
    }
  }

  /**
   * Checks that the given numpy arrays have the same dimensions.
   * Assumes the arrays are 1 dimensional.
   */
  static void check1DNumpyArraysSameDims(const py::buffer_info& buf_1,
                                         const py::buffer_info& buf_2) {
    if (buf_1.shape.at(0) != buf_2.shape.at(0)) {
      std::stringstream ss;
      ss << "[BuilderVector::check1DNumpyArraysSameDims] The given numpy "
            "arrays must have the same dimensions. Found "
         << buf_1.shape.size() << " and " << buf_1.shape.size() << "."
         << std::endl;
      throw std::invalid_argument(ss.str());
    }
  }

  /**
   * Returns the size of the given numpy array.
   * Assumes the array is 1 dimensional.
   */
  static size_t getNumpyArraySize(const py::buffer_info& buf) {
    return buf.shape.at(0);
  }

  /**
   * Extends a vector using the contents of a numpy array.
   */
  template <typename T>
  static void extendVectorByNumpyArray(std::vector<T>& vec,
                                       const py::buffer_info& buf) {
    vec.insert(vec.begin(), static_cast<T*>(buf.ptr),
               static_cast<T*>(buf.ptr) + buf.shape.at(0));
  }
};

/**
 * A concrete implementation of BuilderVector for sparse vectors.
 */
struct SparseBuilderVector : public BuilderVector {
  void addSingleFeature(uint32_t start_dim, float value) final {
    _indices.push_back(start_dim);
    _values.push_back(value);
  }

  void addSparseFeatures(std::vector<uint32_t>& indices,
                         std::vector<float>& values) final {
    _indices.insert(_indices.end(), indices.begin(), indices.end());
    _values.insert(_values.end(), values.begin(), values.end());
  }

  void addSparseFeaturesNumpy(
      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& indices,
      py::array_t<float, py::array::c_style | py::array::forcecast>& values)
      final {
    const py::buffer_info idx_buf = indices.request();
    const py::buffer_info val_buf = values.request();

    checkNumpyArray1D(idx_buf);
    checkNumpyArray1D(val_buf);
    check1DNumpyArraysSameDims(idx_buf, val_buf);

    extendVectorByNumpyArray(_indices, idx_buf);
    extendVectorByNumpyArray(_values, val_buf);
  }

  void addDenseFeatures(uint32_t start_dim, std::vector<float>& values) final {
    for (uint32_t idx = start_dim; idx < start_dim + values.size(); idx++) {
      _indices.push_back(idx);
    }
    _values.insert(_values.end(), values.begin(), values.end());
  }

  void addDenseFeaturesNumpy(
      uint32_t start_dim,
      py::array_t<float, py::array::c_style | py::array::forcecast>& values)
      final {
    const py::buffer_info val_buf = values.request();
    checkNumpyArray1D(val_buf);

    for (uint32_t idx = start_dim; idx < start_dim + getNumpyArraySize(val_buf);
         idx++) {
      _indices.push_back(idx);
    }
    extendVectorByNumpyArray(_values, val_buf);
  }

  void incrementAtIndices(std::vector<uint32_t>& indices, float inc) final {
    std::sort(indices.begin(), indices.end());
    uint32_t impossible = std::numeric_limits<uint32_t>::
        max();  // Way greater than prime mod so no index will be equal to this.
    indices.push_back(impossible);
    uint32_t last_idx = impossible;
    float last_idx_val = 0.0;
    for (uint32_t idx : indices) {
      if (idx != last_idx && last_idx != impossible) {
        addSingleFeature(last_idx, last_idx_val);
        last_idx_val = 0.0;
      }
      last_idx = idx;
      last_idx_val += inc;
    }
  }

  bolt::BoltVector toBoltVector() final {
    // TODO(Geordie): This copies. Is there a better way? Is it necessary to
    // optimize?
    return bolt::BoltVector::makeSparseVector(_indices, _values);
  }

 private:
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
};

/**
 * A concrete implementation of BuilderVector for dense vectors.
 */
struct DenseBuilderVector : public BuilderVector {
  void addSingleFeature(uint32_t start_dim, float value) final {
    checkStartDim(start_dim);
    _values.push_back(value);
  }

  /**
   * A dense vector does not support sparse features.
   */
  void addSparseFeatures(std::vector<uint32_t>& indices,
                         std::vector<float>& values) final {
    (void)indices;
    (void)values;
    throw std::invalid_argument(
        "[DenseBuilderVector::addSparseFeatures] Dense vector does not support "
        "this operation.");
  }

  /**
   * A dense vector does not support sparse features.
   */
  void addSparseFeaturesNumpy(
      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& indices,
      py::array_t<float, py::array::c_style | py::array::forcecast>& values)
      final {
    (void)indices;
    (void)values;

    throw std::invalid_argument(
        "[DenseBuilderVector::addSparseFeatures] Dense vector does not support "
        "this operation.");
  }

  void addDenseFeatures(uint32_t start_dim, std::vector<float>& values) final {
    checkStartDim(start_dim);
    _values.insert(_values.end(), values.begin(), values.end());
  };

  void addDenseFeaturesNumpy(
      uint32_t start_dim,
      py::array_t<float, py::array::c_style | py::array::forcecast>& values)
      final {
    checkStartDim(start_dim);

    const py::buffer_info val_buf = values.request();
    checkNumpyArray1D(val_buf);
    extendVectorByNumpyArray(_values, val_buf);
  }

  void incrementAtIndices(std::vector<uint32_t>& indices, float inc) final {
    (void)indices;
    (void)inc;

    throw std::invalid_argument(
        "[DenseBuilderVector::incrementAtIndices] Dense vector does not "
        "support this operation.");
  }

  bolt::BoltVector toBoltVector() final {
    // TODO(Geordie): This copies. Is there a better way? Is it necessary to
    // optimize?
    return bolt::BoltVector::makeDenseVector(_values);
  };

 private:
  /**
   * Ensures that features are written to the right position in the vector.
   * Since dense vectors only support dense features, new features must
   * start right after previously added features; start_dim must be equal to
   * the current size of the dense vector.
   */
  void checkStartDim(uint32_t start_dim) {
    if (_values.size() != start_dim) {
      std::stringstream ss;
      ss << "[DenseBuilderVector::addDenseFeatures] start_dim (" << start_dim
         << ") is not equal to _values.size() (" << _values.size() << ")"
         << std::endl;
      throw std::invalid_argument(ss.str());
    }
  }

  std::vector<float> _values;
};

}  // namespace thirdai::dataset