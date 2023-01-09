#include "BoltV2TrainPython.h"
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

void createBoltV2TrainSubmodule(py::module_& module) {
  auto train = module.def_submodule("train");

  py::class_<Trainer>(train, "Trainer")
      .def(py::init<nn::model::ModelPtr>(), py::arg("model"))
      .def("train", &Trainer::train, py::arg("train_data"), py::arg("epochs"),
           py::arg("learning_rate"), py::arg("train_metrics"),
           py::arg("validation_data"), py::arg("validation_metrics"),
           py::arg("steps_per_validation"), py::arg("callbacks"));

  auto metrics = train.def_submodule("metrics");

  py::class_<metrics::Metric, metrics::MetricPtr>(metrics, "Metric");  // NOLINT

  py::class_<metrics::LossMetric, std::shared_ptr<metrics::LossMetric>,
             metrics::Metric>(metrics, "LossMetric")
      .def(py::init<nn::loss::LossPtr>(), py::arg("loss_fn"));

  py::class_<metrics::CategoricalAccuracy,
             std::shared_ptr<metrics::CategoricalAccuracy>, metrics::Metric>(
      metrics, "CategoricalAccuracy")
      .def(py::init<>());

  auto callbacks = train.def_submodule("callbacks");

  py::class_<callbacks::Callback, callbacks::CallbackPtr>(callbacks,  // NOLINT
                                                          "Callback");
}

}  // namespace thirdai::bolt::train::python