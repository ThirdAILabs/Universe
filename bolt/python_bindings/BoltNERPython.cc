#include "BoltNerPython.h"
#include "PybindUtils.h"
#include <bolt/src/NER/model/NerBoltModel.h>
#include <bolt/src/NER/model/NerModel.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_set>

namespace thirdai::bolt::python {

void addNERModels(py::module_& module) {
#if THIRDAI_EXPOSE_ALL
  py::class_<NerBackend, std::shared_ptr<NerBackend>>(  // NOLINT
      module, "NerBackend");

  py::class_<NerBoltModel, NerBackend, std::shared_ptr<NerBoltModel>>(
      module, "BoltNerModel")
      .def(py::init<bolt::ModelPtr>(), py::arg("model"));

#endif

  py::class_<NerModel, std::shared_ptr<NerModel>>(module, "NerModel")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<std::shared_ptr<NerBackend>,
                    std::unordered_map<std::string, uint32_t>>(),
           py::arg("model"), py::arg("label_to_tag_map"))
#endif
      .def("train", &NerModel::train, py::arg("train_data"),
           py::arg("learning_rate") = 1e-5, py::arg("epochs") = 5,
           py::arg("batch_size") = 10000,
           py::arg("train_metrics") = std::vector<std::string>{"loss"},
           py::arg("val_data") = nullptr,
           py::arg("val_metrics") = std::vector<std::string>{})
      .def("get_ner_tags", py::arg("tokens"))
      .def("save", &NerModel::save)
      .def_static("load", &NerModel::load, py::arg("filename"))
      .def(thirdai::bolt::python::getPickleFunction<NerModel>());
}

}  // namespace thirdai::bolt::python