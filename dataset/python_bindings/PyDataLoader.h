#include <dataset/src/DataLoader.h>
#include <pybind11/pybind11.h>

namespace thirdai::dataset {

// For explanation of PYBIND11_OVERRIDE_PURE_NAME, see
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
// Basically this allows us to define data loaders in python
class PyDataLoader : public DataLoader {
 public:
  /* Inherit the constructor */
  using DataLoader::DataLoader;

  std::optional<std::vector<std::string>> nextBatch() override {
    PYBIND11_OVERRIDE_PURE_NAME(
        std::optional<std::vector<std::string>>, /* Return type */
        DataLoader,                              /* Parent class */
        "next_batch",                            /* Name of python function */
        nextBatch,                               /* Name of C++ function */
                                                 /* Empty list of arguments */
    );
  }

  std::optional<std::string> getNextLine() override {
    PYBIND11_OVERRIDE_PURE_NAME(std::optional<std::string>, /* Return type */
                                DataLoader,                 /* Parent class */
                                "get_next_line", /* Name of python function */
                                getNextLine,     /* Name of C++ function */
                                                 /* Empty list of arguments */
    );
  }

  std::string resourceName() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::string,     /* Return type */
                                DataLoader,      /* Parent class */
                                "resource_name", /* Name of python function */
                                resourceName,    /* Name of C++ function */
                                                 /* Empty list of arguments */
    );
  }
};

}  // namespace thirdai::dataset
