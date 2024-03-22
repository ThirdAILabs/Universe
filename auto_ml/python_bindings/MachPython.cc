#include "MachPython.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <auto_ml/src/mach/MachConfig.h>
#include <auto_ml/src/mach/MachRetriever.h>
#include <data/src/transformations/SpladeAugmentation.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <pybind11/stl.h>

namespace thirdai::automl::mach::python {

void getTrainOptions(const py::kwargs& kwargs, TrainOptions& options) {
  if (kwargs.contains("batch_size")) {
    options.batch_size = kwargs["batch_size"].cast<size_t>();
  }
  if (kwargs.contains("max_in_memory_batches")) {
    options.max_in_memory_batches =
        kwargs["max_in_memory_batches"].cast<size_t>();
  }
  if (kwargs.contains("verbose")) {
    options.verbose = kwargs["verbose"].cast<bool>();
  }
  options.interrupt_check = bolt::python::CtrlCCheck();
}

bolt::metrics::History wrappedTrain(
    const MachRetrieverPtr& mach, const data::ColumnMapIteratorPtr& data,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const py::kwargs& kwargs) {
  TrainOptions options;
  getTrainOptions(kwargs, options);

  return mach->train(data, learning_rate, epochs, metrics, callbacks, options);
}

bolt::metrics::History wrappedColdStart(
    const MachRetrieverPtr& mach, const data::ColumnMapIteratorPtr& data,
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const py::kwargs& kwargs) {
  ColdStartOptions options;
  getTrainOptions(kwargs, options);
  if (kwargs.contains("variable_length")) {
    options.variable_length =
        kwargs["variable_length"].cast<data::VariableLengthConfig>();
  }
  if (kwargs.contains("splade_config")) {
    options.splade_config = kwargs["splade_config"].cast<data::SpladeConfig>();
  }

  return mach->coldstart(data, strong_cols, weak_cols, learning_rate, epochs,
                         metrics, callbacks, options);
}

void defineMach(py::module_& module) {
  py::class_<MachConfig>(module, "Mach")
      .def(py::init<>())
      .def("build", &MachConfig::build)
      .def("text_col", &MachConfig::textCol, py::arg("col"))
      .def("id_col", &MachConfig::idCol, py::arg("col"))
      .def("tokenizer", &MachConfig::tokenizer, py::arg("tokenizer"))
      .def("contextual_encoding", &MachConfig::contextualEncoding,
           py::arg("encoding"))
      .def("lowercase", &MachConfig::lowercase, py::arg("lowercase") = true)
      .def("text_feature_dim", &MachConfig::textFeatureDim,
           py::arg("text_feature_dim"))
      .def("emb_dim", &MachConfig::embDim, py::arg("emb_dim"))
      .def("n_buckets", &MachConfig::nBuckets, py::arg("n_bukcets"))
      .def("emb_bias", &MachConfig::embBias, py::arg("bias") = true)
      .def("output_bias", &MachConfig::outputBias, py::arg("bias") = true)
      .def("emb_activation", &MachConfig::embActivation, py::arg("activation"))
      .def("output_activation", &MachConfig::outputActivation,
           py::arg("activation"))
      .def("n_hashes", &MachConfig::nHashes, py::arg("n_hashes"))
      .def("mach_sampling_threshold", &MachConfig::machSamplingThreshold,
           py::arg("threshold"))
      .def("num_buckets_to_eval", &MachConfig::nBucketsToEval,
           py::arg("n_buckets_to_eval"))
      .def("mach_memory_params", &MachConfig::machMemoryParams,
           py::arg("max_memory_ids"), py::arg("max_memory_samples_per_id"))
      .def("freeze_hash_tables_epoch", &MachConfig::freezeHashTablesEpoch,
           py::arg("epoch"));

  py::class_<MachRetriever, MachRetrieverPtr>(module, "MachRetriever")
      .def("train", &wrappedTrain, py::arg("data"), py::arg("learning_rate"),
           py::arg("epochs"), py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{})
      .def("coldstart", &wrappedColdStart, py::arg("data"),
           py::arg("strong_cols"), py::arg("weak_cols"),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{})
      .def("evaluate", &MachRetriever::evaluate, py::arg("data"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("verbose") = true);
}

}  // namespace thirdai::automl::mach::python