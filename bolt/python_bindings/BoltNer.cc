#include "BoltNer.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt/src/NER/data_processor/UnigramDataProcessor.h>
#include <pybind11/stl.h>
#include <memory>

namespace thirdai::bolt::ner::python {
void createNERModule(py::module_& module) {
  auto ner_module = module.def_submodule("ner");

  py::class_<SimpleDataProcessor, std::shared_ptr<SimpleDataProcessor>>(
      ner_module, "SimpleDataProcessor")
      .def(py::init(&SimpleDataProcessor::make), py::arg("tokens_column"),
           py::arg("tags_column"), py::arg("fhr_dim"),
           py::arg("target_word_tokenizers"), py::arg("dyadic_num_intervals"))
      .def("process_token", &SimpleDataProcessor::processToken,
           py::arg("tokens"), py::arg("index"))
      .def("featurize_token_tag_list",
           &SimpleDataProcessor::featurizeTokenTagList, py::arg("tokens"),
           py::arg("tags"));
}
}  // namespace thirdai::bolt::ner::python