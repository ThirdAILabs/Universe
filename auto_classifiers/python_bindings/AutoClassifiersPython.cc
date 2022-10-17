#include "AutoClassifiersPython.h"
#include "BinaryTextClassifier.h"
#include "MultiLabelTextClassifer.h"
#include "TabularClassifier.h"
#include "TextClassifier.h"

namespace thirdai::bolt::python {

template <typename CLASSIFIER>
void defineAutoClassifierCommonMethods(py::class_<CLASSIFIER>& py_class) {
  py_class
      .def("train",
           py::overload_cast<const std::string&, uint32_t, float,
                             std::optional<uint32_t>, std::optional<uint64_t>>(
               &CLASSIFIER::train),
           py::arg("filename"), py::arg("epochs"), py::arg("learning_rate"),
           py::arg("batch_size") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("train",
           py::overload_cast<const std::shared_ptr<dataset::DataLoader>&,
                             uint32_t, float, std::optional<uint64_t>>(
               &CLASSIFIER::train),
           py::arg("data_source"), py::arg("epochs"), py::arg("learning_rate"),
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("evaluate",
           py::overload_cast<const std::string&>(&CLASSIFIER::evaluate),
           py::arg("filename"))
      .def("evaluate",
           py::overload_cast<const std::shared_ptr<dataset::DataLoader>&>(
               &CLASSIFIER::evaluate),
           py::arg("data_source"))
      .def("predict", &CLASSIFIER::predict, py::arg("input"))
      .def("predict_batch", &CLASSIFIER::predictBatch, py::arg("inputs"))
      .def("save", &CLASSIFIER::save, py::arg("filename"))
      .def_static("load", &CLASSIFIER::load, py::arg("filename"));
}

void defineAutoClassifeirsInModule(py::module_& bolt_submodule) {
  /**
   * Text Classifier Definition
   */
  py::class_<TextClassifier> text_classifier(bolt_submodule, "TextClassifier");

  text_classifier.def(
      py::init<uint32_t, uint32_t>(), py::arg("internal_model_dim"),
      py::arg("n_classes"),
      "Constructs a TextClassifier with autotuning.\n"
      "Arguments:\n"
      " * internal_model_dim: int - Specifies the internal dimension used in "
      "the model.\n"
      " * n_classes: int - How many classes or categories are in the "
      "labels of the dataset.\n");

  defineAutoClassifierCommonMethods(text_classifier);

  /**
   * Tabular Classifier Definition
   */
  py::class_<TabularClassifier> tabular_classifier(bolt_submodule,
                                                   "TabularClassifier");

  tabular_classifier.def(
      py::init<uint32_t, uint32_t, std::vector<std::string>>(),
      py::arg("internal_model_dim"), py::arg("n_classes"),
      py::arg("column_datatypes"));

#if THIRDAI_EXPOSE_ALL
  tabular_classifier.def(
      py::init<uint32_t, uint32_t, std::shared_ptr<dataset::TabularMetadata>>(),
      py::arg("internal_model_dim"), py::arg("n_classes"),
      py::arg("tabular_metadata"));
#endif

  defineAutoClassifierCommonMethods(tabular_classifier);

  /**
   * Multi Label Text Classifier Definition
   */
  py::class_<MultiLabelTextClassifier> multi_label_classifier(
      bolt_submodule, "MultiLabelTextClassifier");

  multi_label_classifier
      .def(py::init<uint32_t, float>(), py::arg("n_classes"),
           py::arg("threshold") = 0.95)
      .def("update_threshold", &MultiLabelTextClassifier::updateThreshold);

  defineAutoClassifierCommonMethods(multi_label_classifier);

  /**
   * Binary Text Classifier
   */
  py::class_<BinaryTextClassifier> binary_text_classifier(
      bolt_submodule, "BinaryTextClassifier");

  binary_text_classifier.def(
      py::init<uint32_t, uint32_t, std::optional<float>, bool>(),
      py::arg("n_outputs"), py::arg("internal_model_dim"),
      py::arg("sparsity") = std::nullopt,
      py::arg("use_sparse_inference") = true);

  defineAutoClassifierCommonMethods(binary_text_classifier);
}

}  // namespace thirdai::bolt::python