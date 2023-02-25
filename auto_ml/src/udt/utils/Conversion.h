#pragma once

#include <bolt/src/graph/InferenceOutputTracker.h>
#include <pybind11/numpy.h>

namespace thirdai::automl::udt::utils {

namespace py = pybind11;

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

py::object convertBoltVectorToNumpy(const BoltVector& vector);

py::object convertBoltBatchToNumpy(const BoltBatch& batch);

}  // namespace thirdai::automl::udt::utils