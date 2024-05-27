#pragma once

#include <data/src/ColumnMapIterator.h>
#include <pybind11/pybind11.h>

namespace thirdai::data {

// For explanation of PYBIND11_OVERRIDE_PURE_NAME, see
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
// Basically this allows us to define column map iterators in python
class PyColumnMapIterator : public ColumnMapIterator {
 public:
  /* Inherit the constructor */
  using ColumnMapIterator::ColumnMapIterator;

  std::optional<ColumnMap> next() override {
    PYBIND11_OVERRIDE_PURE_NAME(std::optional<ColumnMap>, /* Return type */
                                ColumnMapIterator,        /* Parent class */
                                "next", /* Name of python function */
                                next,   /* Name of C++ function */
                                        /* Empty list of arguments */
    );
  }

  void restart() override {
    PYBIND11_OVERRIDE_PURE_NAME(void,              /* Return type */
                                ColumnMapIterator, /* Parent class */
                                "restart",         /* Name of python function */
                                restart,           /* Name of C++ function */
                                                   /* Empty list of arguments */
    );
  }

  std::string resourceName() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::string,       /* Return type */
                                ColumnMapIterator, /* Parent class */
                                "resource_name",   /* Name of python function */
                                resourceName,      /* Name of C++ function */
                                                   /* Empty list of arguments */
    );
  }
};

}  // namespace thirdai::data
