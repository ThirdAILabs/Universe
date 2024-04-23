#include "PretrainedBasePython.h"
#include <auto_ml/src/pretrained/PretrainedBase.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace thirdai::automl::python {

void addPretrainedBaseModule(py::module_& module) {
  py::class_<PretrainedBase, std::shared_ptr<PretrainedBase>>(module,
                                                              "PretrainedBase")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<std::string, std::vector<bolt::ModelPtr>,
                    std::vector<data::MachIndexPtr>, dataset::TextTokenizerPtr,
                    bool>(),
           py::arg("input_column"), py::arg("models"), py::arg("indexes"),
           py::arg("tokenizer"), py::arg("lowercase"))
      .def("get_top_hash_buckets", &PretrainedBase::getTopHashBuckets,
           py::arg("phrases"), py::arg("hashes_per_model"))
      .def("get_top_tokens", &PretrainedBase::getTopTokens, py::arg("phrase"),
           py::arg("num_tokens"), py::arg("num_buckets_to_decode"))
      .def("train", &PretrainedBase::train, py::arg("train_data"),
           py::arg("epochs"), py::arg("batch_size"), py::arg("learning_rate"),
           py::arg("val_data"))
#endif
      .def(
          "save",
          [](const PretrainedBasePtr& model, const std::string& filename) {
            return model->save(filename);
          },
          py::arg("filename"))
      .def_static(
          "load",
          [](const std::string& filename) {
            return PretrainedBase::load(filename);
          },
          py::arg("filename"));
}

}  // namespace thirdai::automl::python