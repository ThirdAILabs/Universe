#include "BoltV2TrainPython.h"
#include "PybindUtils.h"
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

void createBoltV2TrainSubmodule(py::module_& module) {
  auto train = module.def_submodule("train");

  // TODO(Nicholas): Add methods to return tensors in data pipeline and remove
  // this.
  train.def("convert_dataset", convertDataset, py::arg("dataset"),
            py::arg("dim"));

  train.def("convert_datasets", convertDatasets, py::arg("datasets"),
            py::arg("dims"));

  py::class_<Trainer>(train, "Trainer")
      .def(py::init<nn::model::ModelPtr>(), py::arg("model"))
      .def("train", &Trainer::train, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs") = 1,
           py::arg("train_metrics") = metrics::InputMetrics(),
           py::arg("validation_data") = std::nullopt,
           py::arg("validation_metrics") = metrics::InputMetrics(),
           py::arg("steps_per_validation") = std::nullopt,
           py::arg("use_sparsity_in_validation") = false,
           py::arg("callbacks") = std::vector<callbacks::CallbackPtr>(),
           bolt::python::OutputRedirect())
      .def("validate", &Trainer::validate, py::arg("validation_data"),
           py::arg("validation_metrics") = metrics::InputMetrics(),
           py::arg("use_sparsity") = false, bolt::python::OutputRedirect());

  auto metrics = train.def_submodule("metrics");

  py::class_<metrics::Metric, metrics::MetricPtr>(metrics, "Metric");  // NOLINT

  py::class_<metrics::LossMetric, std::shared_ptr<metrics::LossMetric>,
             metrics::Metric>(metrics, "LossMetric")
      .def(py::init<nn::loss::LossPtr>(), py::arg("loss_fn"));

  py::class_<metrics::CategoricalAccuracy,
             std::shared_ptr<metrics::CategoricalAccuracy>, metrics::Metric>(
      metrics, "CategoricalAccuracy")
      .def(py::init<nn::autograd::ComputationPtr,
                    nn::autograd::ComputationPtr>(),
           py::arg("outputs"), py::arg("labels"));

  auto callbacks = train.def_submodule("callbacks");

  py::class_<callbacks::Callback, callbacks::CallbackPtr>(callbacks,  // NOLINT
                                                          "Callback");
}

}  // namespace thirdai::bolt::train::python