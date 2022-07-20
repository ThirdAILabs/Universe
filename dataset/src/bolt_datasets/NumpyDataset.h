#pragma once

#include "BoltDatasets.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory.h>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace thirdai::dataset::numpy {

namespace py = pybind11;

template <typename BATCH_T>
class NumpyDataset;

using WrappedNumpyVectors = NumpyDataset<bolt::BoltBatch>;
using WrappedNumpyTokens = NumpyDataset<BoltTokenBatch>;

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// TODO(josh): rewrite this comment
/*
 * The purpose of this class is to make sure that a BoltDataset constructed
 * from a numpy array is memory safe by ensuring that the numpy arrays it is
 * constructed from cannot go out of scope while the dataset is in scope. This
 * problem arrises because if the numpy arrays passed in are not uint32 or
 * float32 then when we cast to that array type a copy will occur. This
 * resulting copy of the array will be a local copy, and thus when the method
 * constructing the dataset returns, the copy will go out of scope and the
 * dataset will be invalidated. This solves that issue.
 */
template <typename BATCH_T>
class NumpyDataset : public InMemoryDataset<BATCH_T> {
 public:
  NumpyDataset(std::vector<BATCH_T>&& batches,
               std::vector<py::buffer_info>&& owning_objects)
      : InMemoryDataset<BATCH_T>(std::move(batches)),
        _owning_objects(std::move(owning_objects)) {}

 private:
  // Not sure if this will work, might need to save NumpyArray objects?
  // Or just use call info
  // Try without to make sure we get an error
  std::vector<py::buffer_info> _owning_objects;
};

inline void printCopyError(const py::str& dtype_recv,
                           const std::string& dtype_expected) {
  std::stringstream stream;
  stream << "Error: array has dtype=" << dtype_recv << " but " << dtype_expected
         << " was expected. Try specifying the dtype of the array or use "
            ".astype(...).";
  throw std::invalid_argument(stream.str());
}

inline bool isTuple(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'tuple'>"));
}

inline bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

inline py::str getDtype(const py::object& obj) {
  return py::str(obj.attr("dtype"));
}

inline bool checkNumpyDtype(const py::object& obj, const std::string& type) {
  return getDtype(obj).equal(py::str(type));
}

inline bool checkNumpyDtypeUint32(const py::object& obj) {
  return checkNumpyDtype(obj, "uint32");
}

inline bool checkNumpyDtypeFloat32(const py::object& obj) {
  return checkNumpyDtype(obj, "float32");
}

inline BoltDatasetPtr denseNumpyToBoltVectorDataset(
    const NumpyArray<float>& examples, uint32_t batch_size) {
  // Get information from examples
  const py::buffer_info examples_buf = examples.request();
  if (examples_buf.shape.size() > 2) {
    throw std::invalid_argument(
        "For now, Numpy dense data must be 2D (each row is a dense data "
        "vector) or 1D (each element is treated as a row).");
  }

  uint64_t num_examples = static_cast<uint64_t>(examples_buf.shape.at(0));

  // If it is a 1D array then we know the dimension is 1.
  uint64_t dimension = examples_buf.shape.size() == 2
                           ? static_cast<uint64_t>(examples_buf.shape.at(1))
                           : 1;
  float* examples_raw_data = static_cast<float*>(examples_buf.ptr);

  // Build batches

  uint64_t num_batches = (num_examples + batch_size - 1) / batch_size;
  std::vector<bolt::BoltBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<bolt::BoltVector> batch_vectors;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      batch_vectors.emplace_back(
          nullptr, examples_raw_data + dimension * vec_idx, nullptr, dimension);
    }

    batches.emplace_back(std::move(batch_vectors));
  }

  // py::buffer_info has its copy constructor and assignment deleted, and I
  // found this was the only way to build the vector without an error
  std::vector<py::buffer_info> owning_objects = {};
  owning_objects.push_back(examples.request());

  return std::make_shared<WrappedNumpyVectors>(std::move(batches),
                                               std::move(owning_objects));
}

template <bool CONVERT_TO_VECTORS>
inline std::conditional_t<CONVERT_TO_VECTORS, BoltDatasetPtr,
                          BoltTokenDatasetPtr>
numpyTokensToBoltDataset(const NumpyArray<uint32_t>& tokens,
                         uint32_t batch_size) {
  const py::buffer_info tokens_buf = tokens.request();

  auto shape = tokens_buf.shape;
  if (shape.size() != 1 && shape.size() != 2) {
    throw std::invalid_argument("Expected 1D or 2D array of tokens.");
  }

  uint64_t total_num_tokens = tokens_buf.shape.at(0);
  uint64_t tokens_per_vector = shape.size() == 1 ? 1 : shape.at(1);
  uint64_t num_vectors = total_num_tokens / tokens_per_vector;

  uint64_t num_batches = (num_vectors + batch_size - 1) / batch_size;

  const uint32_t* token_raw_data = static_cast<const uint32_t*>(tokens_buf.ptr);

  std::vector<
      std::conditional_t<CONVERT_TO_VECTORS, bolt::BoltBatch, BoltTokenBatch>>
      batches;

  for (uint64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<std::conditional_t<CONVERT_TO_VECTORS, bolt::BoltVector,
                                   std::vector<uint32_t>>>
        current_token_batch;

    uint64_t start_vector = batch_idx * batch_size;
    uint64_t end_vector =
        std::min<uint32_t>(num_vectors, start_vector + batch_size);

    for (uint64_t vector_id = start_vector; vector_id < end_vector;
         vector_id++) {
      std::vector<uint32_t> vec_tokens(
          &token_raw_data[vector_id * tokens_per_vector],
          &token_raw_data[(vector_id + 1) * tokens_per_vector]);

      if constexpr (CONVERT_TO_VECTORS) {
        std::vector<float> vec_activations(tokens_per_vector, 1.0);
        current_token_batch.push_back(
            bolt::BoltVector::makeSparseVector(vec_tokens, vec_activations));
      } else {
        current_token_batch.push_back(std::move(vec_tokens));
      }
    }

    batches.emplace_back(std::move(current_token_batch));
  }

  // Since we only do copies we don't need to worry about owning objects
  std::vector<py::buffer_info> owning_objects = {};

  if constexpr (CONVERT_TO_VECTORS) {
    return std::make_shared<WrappedNumpyVectors>(std::move(batches),
                                                 std::move(owning_objects));
  } else {
    return std::make_shared<WrappedNumpyTokens>(std::move(batches),
                                                std::move(owning_objects));
  }
}

inline BoltDatasetPtr numpyToBoltVectorDataset(const py::object& data,
                                               uint32_t batch_size) {
  if (isNumpyArray(data)) {
    if (checkNumpyDtypeFloat32(data)) {
      return denseNumpyToBoltVectorDataset(data.cast<NumpyArray<float>>(),
                                           batch_size);
    }
    if (checkNumpyDtypeUint32(data)) {
      return numpyTokensToBoltDataset<true>(data.cast<NumpyArray<uint32_t>>(),
                                            batch_size);
    }
  }

  throw std::invalid_argument(
      "Expected a numpy array of type uint32 or float32, received " +
      py::str(data.get_type()).cast<std::string>());
}

inline BoltTokenDatasetPtr numpyToBoltTokenDataset(const py::object& data,
                                                   uint32_t batch_size) {
  if (isNumpyArray(data) && checkNumpyDtypeUint32(data)) {
    return numpyTokensToBoltDataset<false>(data.cast<NumpyArray<uint32_t>>(),
                                           batch_size);
  }

  throw std::invalid_argument(
      "Expected a numpy array of type uint32, received " +
      py::str(data.get_type()).cast<std::string>());
}

}  // namespace thirdai::dataset::numpy