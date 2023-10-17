#include "BoltSeismic.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt/src/seismic/SeismicEmbeddingModel.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <pybind11/stl.h>

namespace thirdai::bolt::seismic::python {

void createSeismicSubmodule(py::module_& module) {
  auto seismic = module.def_submodule("seismic");

  py::class_<SubcubeMetadata>(seismic, "SubcubeMetadata")
      .def(py::init<std::string, size_t, size_t, size_t>(), py::arg("volume"),
           py::arg("x"), py::arg("y"), py::arg("z"));

  py::class_<SeismicEmbeddingModel, std::shared_ptr<SeismicEmbeddingModel>>(
      seismic, "SeismicEmbeddingModel")
      .def(py::init<size_t, size_t, size_t, const std::string&,
                    std::optional<size_t>>(),
           py::arg("subcube_shape"), py::arg("patch_shape"),
           py::arg("embedding_dim"), py::arg("size") = "large",
           py::arg("max_pool") = std::nullopt)
      .def("train_on_patches", &SeismicEmbeddingModel::trainOnPatches,
           py::arg("subcubes"), py::arg("subcube_metadata"),
           py::arg("learning_rate"), py::arg("batch_size"),
           py::arg("callbacks"), py::arg("log_interval"),
           py::arg("comm") = std::nullopt)
      .def("embeddings_for_patches",
           &SeismicEmbeddingModel::embeddingsForPatches, py::arg("subcubes"))
      .def_property_readonly("subcube_shape",
                             &SeismicEmbeddingModel::subcubeShape)
      .def_property_readonly("patch_shape", &SeismicEmbeddingModel::patchShape)
      .def_property_readonly("max_pool", &SeismicEmbeddingModel::maxPool)
      .def_property("model", &SeismicEmbeddingModel::getModel,
                    &SeismicEmbeddingModel::setModel)
      .def("save", &SeismicEmbeddingModel::save, py::arg("filename"))
      .def_static("load", &SeismicEmbeddingModel::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<SeismicEmbeddingModel>());

  py::class_<seismic::Checkpoint, std::shared_ptr<seismic::Checkpoint>,
             callbacks::Callback>(seismic, "Checkpoint")
      .def(py::init<std::shared_ptr<SeismicEmbeddingModel>, std::string,
                    size_t>(),
           py::arg("seismic_model"), py::arg("checkpoint_dir"),
           py::arg("interval"));

#if THIRDAI_EXPOSE_ALL
  seismic.def("seismic_labels", &seismicLabels, py::arg("volume"),
              py::arg("x_coord"), py::arg("y_coord"), py::arg("z_coord"),
              py::arg("subcube_shape"), py::arg("label_cube_shape"),
              py::arg("max_label"));
#endif
}

}  // namespace thirdai::bolt::seismic::python