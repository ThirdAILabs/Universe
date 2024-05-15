#include "BoltNERPython.h"
#include "PybindUtils.h"
#include <bolt/src/NER/model/NER.h>
#include <bolt/src/NER/model/NerPretrainedModel.h>
#include <bolt/src/NER/model/NerUDTModel.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>

namespace thirdai::bolt::python {

void addNERModels(py::module_& module) {
#if THIRDAI_EXPOSE_ALL
  py::class_<NerBackend, std::shared_ptr<NerBackend>>(  // NOLINT
      module, "NerBackend");

  py::class_<NerPretrainedModel, NerBackend,
             std::shared_ptr<NerPretrainedModel>>(module, "NerPretrainedModel")
      .def(
          py::init<bolt::ModelPtr, std::unordered_map<std::string, uint32_t>>(),
          py::arg("model"), py::arg("tag_to_label"));

  py::class_<NerUDTModel, NerBackend, std::shared_ptr<NerUDTModel>>(
      module, "NerUDTModel")
      .def(py::init<bolt::ModelPtr, std::string, std::string,
                    std::unordered_map<std::string, uint32_t>,
                    std::vector<dataset::TextTokenizerPtr>>(),
           py::arg("model"), py ::arg("tokens_column"), py::arg("tags_column"),
           py::arg("tag_to_label"),
           py::arg("target_word_tokenizers") =
               std::vector<dataset::TextTokenizerPtr>(
                   {std::make_shared<dataset::NaiveSplitTokenizer>(),
                    std::make_shared<dataset::CharKGramTokenizer>(4)}));
#endif

  py::class_<NER, std::shared_ptr<NER>>(module, "NER")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<std::shared_ptr<NerBackend>>(), py::arg("model"))
#endif
      .def(py::init<std::string, std::string,
                    std::unordered_map<std::string, uint32_t>,
                    std::optional<std::string>,
                    std::vector<dataset::TextTokenizerPtr>>(),
           py ::arg("tokens_column"), py::arg("tags_column"),
           py::arg("tag_to_label"),
           py::arg("pretrained_model_path") = std::nullopt,
           py::arg("target_word_tokenizers") =
               std::vector<dataset::TextTokenizerPtr>(
                   {std::make_shared<dataset::NaiveSplitTokenizer>(),
                    std::make_shared<dataset::CharKGramTokenizer>(4)}))
      .def("train", &NER::train, py::arg("train_data"),
           py::arg("learning_rate") = 1e-5, py::arg("epochs") = 5,
           py::arg("batch_size") = 2000,
           py::arg("train_metrics") = std::vector<std::string>{"loss"},
           py::arg("val_data") = nullptr,
           py::arg("val_metrics") = std::vector<std::string>{})
      .def("get_ner_tags", &NER::getNerTags, py::arg("tokens"),
           py::arg("top_k") = 1)
      .def("save", &NER::save)
      .def_static("load", &NER::load, py::arg("filename"))
      .def(thirdai::bolt::python::getPickleFunction<NER>());
}

}  // namespace thirdai::bolt::python