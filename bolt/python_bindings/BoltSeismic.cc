#include "BoltSeismic.h"
#include <bolt/python_bindings/PybindUtils.h>
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
      .def(py::init<size_t, size_t, size_t, std::optional<size_t>>(),
           py::arg("subcube_shape"), py::arg("patch_shape"),
           py::arg("embedding_dim"), py::arg("max_pool") = std::nullopt)
      .def("train_on_patches", &SeismicModel::trainOnPatches,
           py::arg("subcubes"), py::arg("subcube_metadata"),
           py::arg("learning_rate"), py::arg("batch_size"),
           py::arg("comm") = std::nullopt)
      .def("embeddings_for_patches", &SeismicModel::embeddingsForPatches,
           py::arg("subcubes"))
      .def_property_readonly("subcube_shape", &SeismicModel::subcubeShape)
      .def_property_readonly("patch_shape", &SeismicModel::patchShape)
      .def_property_readonly("max_pool", &SeismicModel::maxPool)
      .def_property("model", &SeismicModel::getModel, &SeismicModel::setModel)
      .def("save", &SeismicModel::save, py::arg("filename"))
      .def_static("load", &SeismicModel::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<SeismicModel>());

#if THIRDAI_EXPOSE_ALL
  seismic.def("seismic_labels", &seismicLabels, py::arg("volume"),
              py::arg("x_coord"), py::arg("y_coord"), py::arg("z_coord"),
              py::arg("subcube_shape"), py::arg("label_cube_shape"),
              py::arg("max_label"));
#endif
}

}  // namespace thirdai::bolt::seismic::python