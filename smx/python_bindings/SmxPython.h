#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::smx::python {

void createSmxSubmodule(py::module_& mod);

}  // namespace thirdai::smx::python