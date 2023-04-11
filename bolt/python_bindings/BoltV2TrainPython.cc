#include "BoltV2TrainPython.h"
#include "PybindUtils.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/callbacks/ReduceLROnPlateau.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
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
    auto [grads, flattened_dim] = _model->getGradients();

    py::capsule free_when_done(
        grads, [](void* ptr) { delete static_cast<float*>(ptr); });

    return NumpyArray(flattened_dim, grads, free_when_done);
  }

  void setGradients(NumpyArray& new_grads) {
    if (new_grads.ndim() != 1) {
      throw std::invalid_argument("Expected grads to be flattened.");
    }

    uint64_t flattened_dim = new_grads.shape(0);
    _model->setGradients(new_grads.data(), flattened_dim);
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
      .def("train", &Trainer::train_with_metric_names, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs") = 1,
           py::arg("train_metrics") = std::vector<std::string>(),
           py::arg("validation_data") = std::nullopt,
           py::arg("validation_metrics") = std::vector<std::string>(),
           py::arg("steps_per_validation") = std::nullopt,
           py::arg("use_sparsity_in_validation") = false,
           py::arg("callbacks") = std::vector<callbacks::CallbackPtr>(),
           bolt::python::OutputRedirect())
      .def("validate", &Trainer::validate, py::arg("validation_data"),
           py::arg("validation_metrics") = metrics::InputMetrics(),
           py::arg("use_sparsity") = false, bolt::python::OutputRedirect())
      .def("validate", &Trainer::validate_with_metric_names,
           py::arg("validation_data"),
           py::arg("validation_metrics") = std::vector<std::string>(),
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
           bolt::python::OutputRedirect())
      .def("should_save_optimizer",
           &DistributedTrainingWrapper::saveWithOptimizer,
           py::arg("should_save_optimizer"));

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

  py::class_<metrics::PrecisionAtK, std::shared_ptr<metrics::PrecisionAtK>,
             metrics::Metric>(metrics, "PrecisionAtK")
      .def(py::init<nn::autograd::ComputationPtr, nn::autograd::ComputationPtr,
                    uint32_t>(),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));

  py::class_<metrics::RecallAtK, std::shared_ptr<metrics::RecallAtK>,
             metrics::Metric>(metrics, "RecallAtK")
      .def(py::init<nn::autograd::ComputationPtr, nn::autograd::ComputationPtr,
                    uint32_t>(),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));

  auto callbacks = train.def_submodule("callbacks");

  py::class_<callbacks::Callback, callbacks::CallbackPtr>(callbacks,  // NOLINT
                                                          "Callback");

  py::class_<callbacks::ReduceLROnPlateau,
             std::shared_ptr<callbacks::ReduceLROnPlateau>,
             callbacks::Callback>(callbacks, "ReduceLROnPlateau")
      .def(py::init<std::string, uint32_t, uint32_t, float, float, bool, bool,
                    float>(),
           py::arg("metric"), py::arg("patience") = 10, py::arg("cooldown") = 0,
           py::arg("decay_factor") = 0.1, py::arg("threshold") = 1e-3,
           py::arg("relative_threshold") = true, py::arg("maximize") = true,
           py::arg("min_lr") = 0);
}

}  // namespace thirdai::bolt::train::python