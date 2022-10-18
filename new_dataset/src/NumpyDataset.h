#pragma once

#include "Dataset.h"
#include <bolt_vector/src/BoltVector.h>
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

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

class NumpyDataset final : public Dataset {
 public:
  NumpyDataset(std::vector<BoltVector>&& vectors,
               std::vector<py::object>&& objects_to_keep_alive)
      : Dataset(std::move(vectors)),
        _objects_to_keep_alive(std::move(objects_to_keep_alive)) {}

 private:
  std::vector<py::object> _objects_to_keep_alive;
};

using NumpyDatasetPtr = std::shared_ptr<NumpyDataset>;

inline DatasetPtr denseNumpyToDataset(const NumpyArray<float>& examples) {
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

  std::vector<BoltVector> vectors;

  for (uint64_t vec_idx = 0; vec_idx < num_examples; vec_idx++) {
    vectors.emplace_back(
        /* an = */ nullptr, /* a = */ examples_raw_data + dimension * vec_idx,
        /* g = */ nullptr, /* l = */ dimension);
  }

  std::vector<py::object> objects_to_keep_alive = {examples};

  return std::make_shared<NumpyDataset>(std::move(vectors),
                                        std::move(objects_to_keep_alive));
}

}  // namespace thirdai::dataset::numpy