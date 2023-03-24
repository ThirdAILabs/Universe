#include "BoltV2TrainPython.h"
#include "PybindUtils.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/DistributedTrainingWrapper.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

class GradientReference {
 public:
  explicit GradientReference(DistributedTrainingWrapperPtr model)
      : _model(std::move(model)) {}

  using NumpyArray =
      py::array_t<float, py::array::c_style | py::array::forcecast>;

  NumpyArray getGradients() const {
    auto grad_ref = _model->getGradients();

    py::capsule free_when_done(
        grad_ref.data, [](void* ptr) { delete static_cast<float*>(ptr); });

    return NumpyArray(grad_ref.flattened_dim, grad_ref.data, free_when_done);
  }

  void setGradients(NumpyArray& new_grads) {
    if (new_grads.ndim() != 1) {
      throw std::invalid_argument("Expected grads to be flattened.");
    }

    uint64_t flattened_dim = new_grads.shape(0);
    _model->setGradents({new_grads.mutable_data(), flattened_dim});
  }

 private:
  DistributedTrainingWrapperPtr _model;
};

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

  py::class_<GradientReference>(train, "GradientReference")
      .def("get_gradients", &GradientReference::getGradients)
      .def("set_gradients", &GradientReference::setGradients,
           py::arg("flattened_gradients"));

  py::class_<DistributedTrainingWrapper, DistributedTrainingWrapperPtr>(
      train, "DistributedTrainingWrapper")
      .def(py::init<const nn::model::ModelPtr&, const TrainConfig&, uint32_t>(),
           py::arg("model"), py::arg("train_config"), py::arg("worker_id"))
      .def("compute_and_store_batch_gradients",
           &DistributedTrainingWrapper::computeAndStoreBatchGradients,
           py::arg("batch_idx"))
      .def("update_parameters", &DistributedTrainingWrapper::updateParameters)
      .def("num_batches", &DistributedTrainingWrapper::numBatches)
      .def("set_datasets", &DistributedTrainingWrapper::setDatasets,
           py::arg("train_data"), py::arg("train_labels"))
      .def("finish_training", &DistributedTrainingWrapper::finishTraining, "")
      .def_property_readonly(
          "model",
          [](DistributedTrainingWrapper& wrapped_model) {
            return wrapped_model.getModel();
          },
          py::return_value_policy::reference_internal)
      .def("freeze_hash_tables", &DistributedTrainingWrapper::freezeHashTables,
           py::arg("insert_labels_if_not_found"))
      .def(
          "gradient_reference",
          [](DistributedTrainingWrapperPtr& model) {
            return GradientReference(model);
          },
          py::return_value_policy::reference_internal)
      .def("get_updated_metrics",
           &DistributedTrainingWrapper::getTrainingMetrics,
           bolt::python::OutputRedirect())
      .def("validate_and_save_if_best",
           &DistributedTrainingWrapper::validationAndSaveBest,
           bolt::python::OutputRedirect());

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