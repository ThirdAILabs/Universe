#include "BoltPython.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <dataset/src/DataSource.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& bolt_submodule) {
  py::class_<TrainConfig, TrainConfigPtr>(bolt_submodule, "TrainConfig")
      .def(py::init(&TrainConfig::makeConfig), py::arg("learning_rate"),
           py::arg("epochs"))
      .def("with_metrics", &TrainConfig::withMetrics, py::arg("metrics"))
      .def("silence", &TrainConfig::silence)
#if THIRDAI_EXPOSE_ALL
      // We do not want to expose these methods to customers to hide complexity.
      .def("with_rebuild_hash_tables", &TrainConfig::withRebuildHashTables,
           py::arg("rebuild_hash_tables"))
      .def("with_reconstruct_hash_functions",
           &TrainConfig::withReconstructHashFunctions,
           py::arg("reconstruct_hash_functions"))
      // We do not want to expose this method because it will not work correctly
      // with the ModelPipeline since it won't sae the entire pipeline.
      .def("with_save_parameters", &TrainConfig::withSaveParameters,
           py::arg("save_prefix"), py::arg("save_frequency"))
#endif
      .def("with_callbacks", &TrainConfig::withCallbacks, py::arg("callbacks"))
      .def("with_validation", &TrainConfig::withValidation,
           py::arg("validation_data"), py::arg("validation_labels"),
           py::arg("eval_config"), py::arg("validation_frequency") = 0,
           py::arg("save_best_per_metric") = "",
           R"pbdoc(
Add validation options to execute validation during training. Can be used to
configure input data and labels, frequency to validate and optionally saving
best model per a specified metric.

Args:
    validation_data (dataset.BoltDataset): 
        Input dataset for validation
    validation_label (dataset.BoltDataset): 
        Ground truth labels to use during validation
    eval_config (bolt.EvalConfig): 
        See EvalConfig.
    validation_frequency (int, optional): 
        Interval of updates (batches) to run validation and report
        metrics. Defaults to 0, which is no validation amidst
        training.
    save_best_per_metric (str, optional): 
        Whether to save best model based on validation. Needs
        with_save_parameters(...) configured.  Defaults to empty
        string, which implies no saving best model. Note that this requires the
        tracked metric to be configured via `with_metrics(...)`.

)pbdoc")
      .def_property_readonly(
          "num_epochs", [](TrainConfig& config) { return config.epochs(); },
          "Returns the number of epochs a model with this TrainConfig will "
          "train for.")
      .def_property_readonly(
          "learning_rate",
          [](TrainConfig& config) { return config.learningRate(); },
          "Returns the learning rate a model with this TrainConfig will train "
          "with.")
      .def(getPickleFunction<TrainConfig>())
      .def("with_log_loss_frequency", &TrainConfig::withLogLossFrequency,
           py::arg("log_loss_frequency"));

  py::class_<EvalConfig>(bolt_submodule, "EvalConfig")
      .def(py::init(&EvalConfig::makeConfig))
      .def("enable_sparse_inference", &EvalConfig::enableSparseInference)
      .def("with_metrics", &EvalConfig::withMetrics, py::arg("metrics"))
      .def("silence", &EvalConfig::silence)
      .def("return_activations", &EvalConfig::returnActivations);
}

}  // namespace thirdai::bolt::python
