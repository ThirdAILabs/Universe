#include "MachPretrainedPython.h"
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <auto_ml/src/pretrained/MachPretrained.h>

namespace thirdai::automl::python {

    void addMachPretrainedModule(py::module_& module) {
        py::class_<MachPretrained, std::shared_ptr<MachPretrained>>(
            module, "MachPretrained"
        )
        #if THIRDAI_EXPOSE_ALL
        .def(py::init<std::string, std::vector<bolt::ModelPtr>, std::vector<data::MachIndexPtr>,
                    dataset::TextTokenizerPtr, uint32_t>(), py::arg("input_column"), py::arg("models"),
            py::arg("indexes"), py::arg("tokenizer"), py::arg("vocab_size"))
        .def("get_top_hash_buckets", &MachPretrained::getTopHashBuckets, py::arg("phrases"),
        py::arg("hashes_per_model"))
        .def("get_top_tokens", &MachPretrained::getTopTokens, py::arg("phrase"), py::arg("num_tokens"),
        py::arg("num_buckets_to_decode"))
        #endif
        .def("train", &MachPretrained::train, py::arg("train_data"), py::arg("epochs"),
        py::arg("batch_size"), py::arg("learning_rate"), py::arg("val_data"))
        .def("save",
          [](const MachPretrainedPtr& model, const std::string& filename) {
            return model->save(filename);
          }, py::arg("filename"))
        .def_static("load", [](const std::string& filename) {
            return MachPretrained::load(filename);
          }, py::arg("filename"));
    }

} // namespace thirdai::automl::python