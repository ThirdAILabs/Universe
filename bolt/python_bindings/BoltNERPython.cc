#include "BoltNERPython.h"
#include "PybindUtils.h"
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/NER/model/NerBackend.h>
#include <bolt/src/NER/model/NerBoltModel.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/ner/NerDyadicDataProcessor.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

namespace thirdai::bolt::NER::python {

void addNERModels(py::module_& module) {
#if THIRDAI_EXPOSE_ALL
  py::class_<NerModelInterface, std::shared_ptr<NerModelInterface>>(  // NOLINT
      module, "NerBackend");

  py::class_<NerBoltModel, NerModelInterface, std::shared_ptr<NerBoltModel>>(
      module, "NerBoltModel")
      .def(py::init<bolt::ModelPtr, std::string, std::string,
                    std::unordered_map<std::string, uint32_t>>(),
           py::arg("model"), py ::arg("tokens_column"), py::arg("tags_column"),
           py::arg("tag_to_label"));
#endif

  py::class_<NER, std::shared_ptr<NER>>(module, "NER")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<std::shared_ptr<NerModelInterface>>(), py::arg("model"))
      .def("_get_model", &NER::getModel)
#endif
      .def_static(
          "from_pretrained",
          [](const std::string& model_path, const std::string& tokens_column,
             const std::string& tags_column,
             const std::unordered_map<std::string, uint32_t>& tag_to_label) {
            return NER(model_path, tokens_column, tags_column, tag_to_label);
          },
          py::arg("model_path"), py::arg("tokens_column"),
          py::arg("tags_column"), py::arg("tag_to_label"))
      .def("train", &NER::train, py::arg("train_data"),
           py::arg("learning_rate") = 1e-5, py::arg("epochs") = 5,
           py::arg("batch_size") = 2000,
           py::arg("train_metrics") = std::vector<std::string>{"loss"},
           py::arg("val_data") = nullptr,
           py::arg("val_metrics") = std::vector<std::string>{})
      .def("get_ner_tags", &NER::getNerTags, py::arg("tokens"),
           py::arg("top_k") = 1)
      .def("save", &NER::save)
      .def("type", &NER::type)
      .def("tags_column", &NER::getTagsColumn)
      .def("tokens_column", &NER::getTokensColumn)
      .def_static("load", &NER::load, py::arg("filename"))
      .def(thirdai::bolt::python::getPickleFunction<NER>());
}

}  // namespace thirdai::bolt::NER::python