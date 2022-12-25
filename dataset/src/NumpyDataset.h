#pragma once

#include "InMemoryDataset.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace thirdai::dataset::numpy {

namespace py = pybind11;

template <typename BATCH_T>
class NumpyDataset;

using WrappedNumpyVectors = NumpyDataset<BoltBatch>;

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// TODO(Josh/Nick): Some of this code is currently duplicated in new_dataset,
// but when we deprecate the current dataset API we can delete it and fix the
// duplication.

/**
 * This class is an InMemoryDataset that is backed by numpy arrays. The
 * batches that are passed into the constructor are assumed to be shallow
 * objects on top of memory owned by the py::objects in the vector
 * objects_to_keep_alive. This class does NOT manager the memory of those
 * py::objects manually. Instead, merely having a copy of the py::objects
 * increments the reference counter and prevents python from deleting those
 * objects until this object is deleted.
 */
template <typename BATCH_T>
class NumpyDataset final : public InMemoryDataset<BATCH_T> {
 public:
  NumpyDataset(std::vector<BATCH_T>&& batches,
               std::vector<py::object>&& objects_to_keep_alive)
      : InMemoryDataset<BATCH_T>(std::move(batches)),
        _objects_to_keep_alive(std::move(objects_to_keep_alive)) {}

 private:
  std::vector<py::object> _objects_to_keep_alive;
};

inline bool isTuple(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'tuple'>"));
}

inline bool isNumpyArray(const py::object& obj) {
  return py::str(obj.get_type()).equal(py::str("<class 'numpy.ndarray'>"));
}

inline py::str getNumpyDtype(const py::object& obj) {
  return py::str(obj.attr("dtype"));
}

inline bool checkNumpyDtype(const py::object& obj, const std::string& type) {
  return getNumpyDtype(obj).equal(py::str(type));
}

inline bool isNumpyUint32(const py::object& obj) {
  return checkNumpyDtype(obj, "uint32");
}

inline bool isNumpyFloat32(const py::object& obj) {
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
  std::vector<BoltBatch> batches;

  for (uint32_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_vectors;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      batch_vectors.emplace_back(
          nullptr, examples_raw_data + dimension * vec_idx, nullptr, dimension);
    }

    batches.emplace_back(std::move(batch_vectors));
  }

  std::vector<py::object> objects_to_keep_alive = {examples};

  return std::make_shared<WrappedNumpyVectors>(
      std::move(batches), std::move(objects_to_keep_alive));
}

inline BoltDatasetPtr numpyTokensToBoltDataset(
    const NumpyArray<uint32_t>& tokens, uint64_t batch_size) {
  const py::buffer_info tokens_buf = tokens.request();

  if (tokens.ndim() != 1 && tokens.ndim() != 2) {
    throw std::invalid_argument("Expected 1D or 2D array of tokens.");
  }

  uint64_t tokens_per_vector = tokens.ndim() == 1 ? 1 : tokens.shape(1);
  uint64_t num_vectors = tokens.shape(0);

  uint64_t num_batches = (num_vectors + batch_size - 1) / batch_size;

  const uint32_t* token_raw_data = static_cast<const uint32_t*>(tokens_buf.ptr);

  std::vector<BoltBatch> batches;

  for (uint64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> current_token_batch;

    uint64_t start_vector = batch_idx * batch_size;
    uint64_t end_vector =
        std::min<uint64_t>(num_vectors, start_vector + batch_size);

    for (uint64_t vector_id = start_vector; vector_id < end_vector;
         vector_id++) {
      std::vector<uint32_t> vec_tokens(
          &token_raw_data[vector_id * tokens_per_vector],
          &token_raw_data[(vector_id + 1) * tokens_per_vector]);

      std::vector<float> vec_activations(tokens_per_vector, 1.0);
      current_token_batch.push_back(BoltVector::sparse(
          vec_tokens, vec_activations, /*has_gradient=*/false));
    }

    batches.emplace_back(std::move(current_token_batch));
  }

  // Since we only do copies we don't need to worry about owning objects
  std::vector<py::object> objects_to_keep_alive = {};

  return std::make_shared<WrappedNumpyVectors>(
      std::move(batches), std::move(objects_to_keep_alive));
}

inline void verifySparseNumpyTuple(const py::tuple& tup) {
  if (tup.size() != 3) {
    throw std::invalid_argument(
        "If passing in a tuple to specify a sparse dataset, "
        "you must pass in a tuple of 3 arrays (indices, values, offsets), "
        "but you passed in a tuple of length: " +
        std::to_string(tup.size()));
  }

  if (!isNumpyArray(tup[0]) || !isNumpyArray(tup[1]) || !isNumpyArray(tup[2])) {
    throw std::invalid_argument(
        "If passing in a tuple to specify a sparse dataset, the tuple must be "
        "of 3 numpy arrays (indices, values, offsets), but you passed in a non "
        "numpy array for one of the tuple elements.");
  }

  if (!isNumpyUint32(tup[0])) {
    throw std::invalid_argument(
        "The first element of a tuple for conversion must be a uint32 numpy "
        "array");
  }
  if (!isNumpyFloat32(tup[1])) {
    throw std::invalid_argument(
        "The second element of a tuple for conversion must be a float32 numpy "
        "array");
  }
  if (!isNumpyUint32(tup[2])) {
    throw std::invalid_argument(
        "The third element of a tuple for conversion must be a uint32 numpy "
        "array");
  }
}

inline BoltDatasetPtr numpyArraysToSparseBoltDataset(
    const NumpyArray<uint32_t>& indices, const NumpyArray<float>& values,
    const NumpyArray<uint32_t>& offsets, uint64_t batch_size) {
  if (offsets.shape(0) == 0) {
    throw std::invalid_argument(
        "Offsets array must be at least of size 1, since it has n + 1 entries "
        "(where n is the number of vectors in the sparse dataset)");
  }
  uint64_t num_examples = static_cast<uint64_t>(offsets.shape(0) - 1);

  uint32_t* offsets_raw_data = const_cast<uint32_t*>(offsets.data());
  uint32_t total_data_size = offsets_raw_data[num_examples];

  if (total_data_size != indices.shape(0)) {
    throw std::invalid_argument(
        "According to the offsets array, the flattened indices array should be "
        "of length " +
        std::to_string(total_data_size) + ", but it is actually of length " +
        std::to_string(indices.shape(0)));
  }
  uint32_t* indices_raw_data = const_cast<uint32_t*>(indices.data());

  if (total_data_size != values.shape(0)) {
    throw std::invalid_argument(
        "According to the offsets array, the flattened values array should be "
        "of length " +
        std::to_string(total_data_size) + ", but it is actually of length " +
        std::to_string(values.shape(0)));
  }
  float* values_raw_data = const_cast<float*>(values.data());

  // Build batches

  uint64_t num_batches = (num_examples + batch_size - 1) / batch_size;
  std::vector<BoltBatch> batches;

  for (uint64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<BoltVector> batch_vectors;

    uint64_t start_vec_idx = batch_idx * batch_size;
    uint64_t end_vec_idx = std::min(start_vec_idx + batch_size, num_examples);
    for (uint64_t vec_idx = start_vec_idx; vec_idx < end_vec_idx; ++vec_idx) {
      // owns_data = false because we don't want the numpy array to be deleted
      // if this batch (and thus the underlying vectors) get deleted
      auto vector_length =
          offsets_raw_data[vec_idx + 1] - offsets_raw_data[vec_idx];
      batch_vectors.emplace_back(indices_raw_data + offsets_raw_data[vec_idx],
                                 values_raw_data + offsets_raw_data[vec_idx],
                                 nullptr, vector_length);
    }

    batches.emplace_back(std::move(batch_vectors));
  }

  std::vector<py::object> objects_to_keep_alive = {indices, values, offsets};

  return std::make_shared<WrappedNumpyVectors>(
      std::move(batches), std::move(objects_to_keep_alive));
}

inline BoltDatasetPtr tupleToSparseBoltDataset(const py::object& obj,
                                               uint64_t batch_size) {
  py::tuple tup = obj.cast<py::tuple>();
  verifySparseNumpyTuple(tup);

  NumpyArray<uint32_t> indices = tup[0].cast<NumpyArray<uint32_t>>();
  NumpyArray<float> values = tup[1].cast<NumpyArray<float>>();
  NumpyArray<uint32_t> offsets = tup[2].cast<NumpyArray<uint32_t>>();

  return numpyArraysToSparseBoltDataset(indices, values, offsets, batch_size);
}

inline void verifyBatchSize(uint64_t batch_size) {
  if (batch_size == 0) {
    throw std::invalid_argument(
        "Passed in batch size was 0, but must be greater than 0");
  }
}

inline BoltDatasetPtr numpyToBoltVectorDataset(const py::object& data,
                                               uint64_t batch_size) {
  verifyBatchSize(batch_size);
  if (isNumpyArray(data)) {
    if (isNumpyFloat32(data)) {
      return denseNumpyToBoltVectorDataset(data.cast<NumpyArray<float>>(),
                                           batch_size);
    }
    if (isNumpyUint32(data)) {
      return numpyTokensToBoltDataset(data.cast<NumpyArray<uint32_t>>(),
                                      batch_size);
    }

    throw std::invalid_argument(
        "Expected a numpy array of type uint32 or float32 but instead "
        "recieved "
        "a numpy array of type " +
        getNumpyDtype(data).cast<std::string>());
  }

  if (isTuple(data)) {
    return tupleToSparseBoltDataset(data, batch_size);
  }

  throw std::invalid_argument(
      "Expected a numpy array or a tuple of numpy arrays, but instead "
      "received "
      "an object of type " +
      py::str(data.get_type()).cast<std::string>());
}

}  // namespace thirdai::dataset::numpy
