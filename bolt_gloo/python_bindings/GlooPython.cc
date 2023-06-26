#include "GlooPython.h"
#include<bolt_gloo/src/glooGroup.cc>

namespace thirdai::bolt_gloo::python {
    void defineGlooSubmodule(py::module_& module){
        py::class_<glooGroup>(module, "glooGroup")
        .def(py::init<uint32_t , uint32_t, std::string&, std::string&, std::string&>(),
        py::arg("world_size"), py::arg("rank"), py::arg("group_name"), py::arg("store_path"), py::arg("process_ip_address"));
    }
}  // namespace thirdai::bolt_gloo:: 