#include "BoltSeismic.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt/src/seismic/SeismicClassifier.h>
#include <bolt/src/seismic/SeismicEmbedding.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <pybind11/stl.h>

namespace thirdai::bolt::seismic::python {

void createSeismicSubmodule(py::module_& module) {
  auto seismic = module.def_submodule("seismic");

  py::class_<SeismicBase, std::shared_ptr<SeismicBase>>(seismic, "SeismicBase")
      .def("embeddings_for_patches", &SeismicBase::embeddingsForPatches,
           py::arg("subcubes"), py::arg("sparse_inference") = false)
      .def_property_readonly("subcube_shape", &SeismicBase::subcubeShape)
      .def_property_readonly("patch_shape", &SeismicBase::patchShape)
      .def_property_readonly("max_pool", &SeismicBase::maxPool)
      .def_property("model", &SeismicBase::getModel, &SeismicBase::setModel);

  py::class_<SeismicEmbedding, std::shared_ptr<SeismicEmbedding>, SeismicBase>(
      seismic, "SeismicEmbedding")
      .def(py::init(&SeismicEmbedding::makeCube), py::arg("subcube_shape"),
           py::arg("patch_shape"), py::arg("embedding_dim"),
           py::arg("size") = "large", py::arg("max_pool") = std::nullopt)
      .def(py::init(&SeismicEmbedding::make), py::arg("subcube_shape"),
           py::arg("patch_shape"), py::arg("embedding_dim"),
           py::arg("size") = "large", py::arg("max_pool") = std::nullopt)
      .def("train_on_patches", &SeismicEmbedding::trainOnPatches,
           py::arg("subcubes"), py::arg("subcube_metadata"),
           py::arg("learning_rate"), py::arg("batch_size"),
           py::arg("callbacks"), py::arg("log_interval"),
           py::arg("comm") = nullptr)
      .def("forward_finetuning", &SeismicEmbedding::forward,
           py::arg("subcubes"))
      .def("backpropagate_finetuning", &SeismicEmbedding::backpropagate,
           py::arg("gradients"))
      .def("update_parameters", &SeismicEmbedding::updateParameters,
           py::arg("learning_rate"))
      .def("save", &SeismicEmbedding::save, py::arg("filename"))
      .def_static("load", &SeismicEmbedding::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<SeismicEmbedding>());

  py::class_<SeismicClassifier, std::shared_ptr<SeismicClassifier>,
             SeismicBase>(seismic, "SeismicClassifier")
      .def(py::init<const std::shared_ptr<SeismicBase>&, size_t, bool>(),
           py::arg("emb_model"), py::arg("n_classes"),
           py::arg("freeze_emb_model") = false)
      .def("train_on_patches", &SeismicClassifier::trainOnPatches,
           py::arg("subcubes"), py::arg("labels"), py::arg("learning_rate"),
           py::arg("batch_size"), py::arg("callbacks"), py::arg("log_interval"),
           py::arg("comm") = nullptr)
      .def("predictions_for_patches", &SeismicClassifier::predictionsForPatches,
           py::arg("subcubes"), py::arg("sparse_inference") = false)
      .def("save", &SeismicClassifier::save, py::arg("filename"))
      .def_static("load", &SeismicClassifier::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<SeismicClassifier>());

  py::class_<seismic::Checkpoint, std::shared_ptr<seismic::Checkpoint>,
             callbacks::Callback>(seismic, "Checkpoint")
      .def(py::init<std::shared_ptr<SeismicBase>, std::string, size_t>(),
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