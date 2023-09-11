#include "BoltTextGeneration.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/ContextualModel.h>
#include <bolt/src/text_generation/DyadicModel.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <pybind11/stl.h>
#include <unordered_set>

namespace thirdai::bolt::python {

void addTextGenerationModels(py::module_& module) {
#if THIRDAI_EXPOSE_ALL
  py::class_<GenerativeBackend, std::shared_ptr<GenerativeBackend>>(  // NOLINT
      module, "GenerativeBackend");

  py::class_<DyadicModel, GenerativeBackend, std::shared_ptr<DyadicModel>>(
      module, "DyadicModel")
      .def(py::init<bolt::ModelPtr>(), py::arg("model"));

  py::class_<ContextualModel, GenerativeBackend,
             std::shared_ptr<ContextualModel>>(module, "ContextualModel")
      .def(py::init<bolt::ModelPtr, dataset::TextGenerationFeaturizerPtr>(),
           py::arg("model"), py::arg("featurizer"));
#endif

  py::class_<GenerativeModel, std::shared_ptr<GenerativeModel>>(
      module, "GenerativeModel")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<std::shared_ptr<GenerativeBackend>,
                    std::unordered_set<uint32_t>,
                    std::unordered_set<uint32_t>>(),
           py::arg("model"), py::arg("allowed_repeats"),
           py::arg("punctuation_tokens"))
#endif
      .def(py::init(&GenerativeModel::load), py::arg("filename"))
      .def("generate", &GenerativeModel::generate, py::arg("input_tokens"),
           py::arg("n_predictions"), py::arg("beam_width"),
           py::arg("temperature") = std::nullopt)
      .def("save", &GenerativeModel::save);
}

}  // namespace thirdai::bolt::python