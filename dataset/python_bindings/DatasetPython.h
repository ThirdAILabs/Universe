#pragma once

#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/BoltInputBatch.h>
#include <dataset/src/batch_types/DenseBatch.h>
#include <dataset/src/batch_types/SparseBatch.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::dataset::python {

void createDatasetSubmodule(py::module_& module);

InMemoryDataset<ClickThroughBatch> loadClickThroughDataset(
    const std::string& filename, uint32_t batch_size,
    uint32_t num_dense_features, uint32_t num_categorical_features,
    bool sparse_labels);

InMemoryDataset<SparseBatch> loadSVMDataset(const std::string& filename,
                                            uint32_t batch_size);

InMemoryDataset<DenseBatch> loadCSVDataset(const std::string& filename,
                                           uint32_t batch_size,
                                           std::string delimiter);

InMemoryDataset<BoltInputBatch> loadBoltSVMDataset(const std::string& filename,
                                                   uint32_t batch_size);

InMemoryDataset<BoltInputBatch> loadBoltCSVDataset(const std::string& filename,
                                                   uint32_t batch_size,
                                                   std::string delimiter);

// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
// for explanation of why we do py::array::c_style and py::array::forcecast.
// Ensures array is an array of floats in dense row major order.
SparseBatch wrapNumpyIntoSparseData(
    const std::vector<py::array_t<
        float, py::array::c_style | py::array::forcecast>>& sparse_values,
    const std::vector<
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
        sparse_indices,
    uint64_t starting_id);

DenseBatch wrapNumpyIntoDenseBatch(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& data,
    uint64_t starting_id);

InMemoryDataset<DenseBatch> denseInMemoryDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size, uint64_t starting_id);

InMemoryDataset<SparseBatch> sparseInMemoryDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_idxs,
	const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_vals,
	const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_offsets,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_idxs,
	const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_offsets,
    uint32_t batch_size, uint64_t starting_id);

InMemoryDataset<BoltInputBatch> denseBoltDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size);

InMemoryDataset<BoltInputBatch> sparseBoltDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_idxs,
	const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_vals,
	const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_offsets,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_idxs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        y_vals,
	const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_offsets,
    uint32_t batch_size);


}  // namespace thirdai::dataset::python