#include "BoltSeismic.h"
#include <bolt/src/seismic/SeismicLabels.h>
#include <bolt/src/seismic/SeismicModel.h>
#include <pybind11/stl.h>

namespace thirdai::bolt::seismic::python {

void createSeismicSubmodule(py::module_& module) {
  auto seismic = module.def_submodule("seismic");

  py::class_<SubcubeMetadata>(seismic, "SubcubeMetadata")
      .def(py::init<std::string, size_t, size_t, size_t>(), py::arg("volume"),
           py::arg("x"), py::arg("y"), py::arg("z"));

  py::class_<SeismicModel, std::shared_ptr<SeismicModel>>(seismic,
                                                          "SeismicModel")
      .def(py::init<size_t, size_t, size_t>(), py::arg("subcube_shape"),
           py::arg("patch_shape"), py::arg("embedding_dim"))
      .def("train", &SeismicModel::train, py::arg("subcubes"),
           py::arg("subcube_metadata"), py::arg("learning_rate"),
           py::arg("batch_size"))
      .def("embeddings", &SeismicModel::embeddings, py::arg("subcubes"))
      .def_property_readonly("subcube_shape", &SeismicModel::subcubeShape)
      .def_property_readonly("patch_shape", &SeismicModel::patchShape)
      .def("save", &SeismicModel::save, py::arg("filename"))
      .def_static("load", &SeismicModel::load, py::arg("filename"));
}

}  // namespace thirdai::bolt::seismic::python