#pragma once

#include <schema/Schema.h>
#include <schema/DynamicCounts.h>
#include <schema/FeatureHashing.h>
#include <schema/Number.h>
#include <schema/NumericalLabel.h>
#include <schema/Text.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::schema::python {

void createSchemaSubmodule(py::module_& module);

}  // namespace thirdai::schema::python
