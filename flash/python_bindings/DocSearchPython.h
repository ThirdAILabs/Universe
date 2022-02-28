#pragma once

#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <flash/src/MaxFlashArray.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;

namespace thirdai::search::python {

using thirdai::dataset::python::wrapNumpyIntoDenseBatch;

// TODO(josh): Make uint8_t configurable, will currently cut off all documents
// at 256 embeddings
class PyMaxFlashArray final : public MaxFlashArray<uint8_t> {
 public:
  PyMaxFlashArray(uint32_t hashes_per_table, uint32_t num_tables,
                  uint32_t dense_input_dimension,
                  uint32_t max_allowable_doc_size)
      : MaxFlashArray<uint8_t>(
            new thirdai::hashing::FastSRP(dense_input_dimension,
                                          hashes_per_table, num_tables),
            hashes_per_table, max_allowable_doc_size) {}

  void addDocument(const py::array_t<float, py::array::c_style |
                                                py::array::forcecast>& data) {
    MaxFlashArray<uint8_t>::addDocument(wrapNumpyIntoDenseBatch(data, 0));
  }

  py::array rankDocuments(
      const py::array_t<float, py::array::c_style | py::array::forcecast>&
          queries,
      const std::vector<uint32_t>& flashes_to_query) {
    auto query_batch = wrapNumpyIntoDenseBatch(queries, 0);
    std::vector<float> sim_sums = MaxFlashArray<uint8_t>::getDocumentScores(
        query_batch, flashes_to_query);

    std::vector<size_t> idx(sim_sums.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(), [&sim_sums](size_t i1, size_t i2) {
      return sim_sums[i1] > sim_sums[i2];
    });

    assert(flashes_to_query.size() == idx.size());
    std::vector<uint32_t> final_result(flashes_to_query.size());
    for (uint32_t i = 0; i < final_result.size(); i++) {
      final_result[i] = flashes_to_query.at(idx[i]);
    }

    auto to_return = py::cast(std::move(final_result));

    return to_return;
  }
};

}  // namespace thirdai::search::python