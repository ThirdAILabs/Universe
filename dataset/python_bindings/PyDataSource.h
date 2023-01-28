#pragma once

#include <dataset/src/DataSource.h>
#include <pybind11/pybind11.h>

namespace thirdai::dataset {

// For explanation of PYBIND11_OVERRIDE_PURE_NAME, see
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
// Basically this allows us to define data sources in python
class PyDataSource : public DataSource {
 public:
  /* Inherit the constructor */
  using DataSource::DataSource;

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        std::optional<std::vector<std::string>>, /* Return type */
        DataSource,                              /* Parent class */
        "next_batch",                            /* Name of python function */
        nextBatch,                               /* Name of C++ function */
        std::ref(target_batch_size)              /* Argument */
    );
  }

  std::optional<std::string> nextLine() override {
    PYBIND11_OVERRIDE_PURE_NAME(std::optional<std::string>, /* Return type */
                                DataSource,                 /* Parent class */
                                "next_line", /* Name of python function */
                                nextLine,    /* Name of C++ function */
                                             /* Empty list of arguments */
    );
  }

  std::string resourceName() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::string,     /* Return type */
                                DataSource,      /* Parent class */
                                "resource_name", /* Name of python function */
                                resourceName,    /* Name of C++ function */
                                                 /* Empty list of arguments */
    );
  }

  void restart() override {
    PYBIND11_OVERRIDE_PURE_NAME(void,       /* Return type */
                                DataSource, /* Parent class */
                                "restart",  /* Name of python function */
                                restart,    /* Name of C++ function */
                                            /* Empty list of arguments */
    );
  }
};

}  // namespace thirdai::dataset
