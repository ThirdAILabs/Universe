// Code to create thirdai modules
#include <bolt/python_bindings/BoltPython.h>
#include <hashing/python_bindings/HashingPython.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <flash/python_bindings/FlashPython.h>
#if THIRDAI_BUILD_SCHEMA
#include <schema/python_bindings/SchemaPython.h>
#endif

// Pybind11 library
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef __clang__
#include <omp.h>
#endif

// TODO(all): Figure out naming convention for python exposed classes and
// methods
// TODO(any): Add docstrings to methods
// TODO(any): Can we remove redudancy in the bindings?
PYBIND11_MODULE(thirdai, m) {  // NOLINT

#ifndef __clang__
  m.def("set_global_num_threads", &omp_set_num_threads,
        py::arg("max_num_threads"));
#endif

  // Per pybind11 docs breaking up the construction of bindings in this way
  // could speed up build times. See below for more info:
  // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time
  thirdai::dataset::python::createDatasetSubmodule(m);

  thirdai::hashing::python::createHashingSubmodule(m);

  thirdai::bolt::python::createBoltSubmodule(m);

  thirdai::search::python::createSearchSubmodule(m);

  #if THIRDAI_BUILD_SCHEMA
  thirdai::schema::python::createSchemaSubmodule(m);
  #endif
}