#include <hashing/src/HashFunction.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <flash/src/Flash.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::search::python {

using thirdai::dataset::python::wrapNumpyIntoDenseBatch;
using thirdai::dataset::python::wrapNumpyIntoSparseData;

void createSearchSubmodule(py::module_& module);

class PyFlash final : public Flash<uint64_t> {
 public:
  explicit PyFlash(const hashing::HashFunction& function)
      : Flash<uint64_t>(function) {}

  PyFlash(const hashing::HashFunction& function, uint32_t reservoir_size)
      : Flash<uint64_t>(function, reservoir_size) {}

  void addDenseBatch(
      const py::array_t<float, py::array::c_style | py::array::forcecast>& data,
      uint64_t starting_id) {
    Flash<uint64_t>::addBatch(wrapNumpyIntoDenseBatch(data, starting_id));
  }

  void addSparseBatch(
      const std::vector<py::array_t<float, py::array::c_style |
                                               py::array::forcecast>>& values,
      const std::vector<
          py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
          indices,
      uint64_t starting_id) {
    Flash<uint64_t>::addBatch(
        wrapNumpyIntoSparseData(values, indices, starting_id));
  }

  py::array queryDenseBatch(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          queries,
      uint32_t top_k) {
    bool pad_zeros = true;
    auto query_batch = wrapNumpyIntoDenseBatch(queries, 0);
    auto result = Flash<uint64_t>::queryBatch(query_batch, top_k, pad_zeros);
    return py::cast(result);
  }

  py::array querySparseBatch(
      const std::vector<py::array_t<
          float, py::array::c_style | py::array::forcecast>>& query_values,
      const std::vector<
          py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
          query_indices,
      uint32_t top_k) {
    bool pad_zeros = true;
    auto query_batch = wrapNumpyIntoSparseData(query_values, query_indices, 0);
    auto result = Flash<uint64_t>::queryBatch(query_batch, top_k, pad_zeros);
    return py::cast(result);
  }
};

}  // namespace thirdai::search::python