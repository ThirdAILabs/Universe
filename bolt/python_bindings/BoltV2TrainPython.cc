#include "BoltV2TrainPython.h"
#include "CtrlCCheck.h"
#include "DistributedCommunicationPython.h"
#include "PyCallback.h"
#include "PybindUtils.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/callbacks/LearningRateScheduler.h>
#include <bolt/src/train/callbacks/Overfitting.h>
#include <bolt/src/train/callbacks/ReduceLROnPlateau.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/FMeasure.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/MachPrecision.h>
#include <bolt/src/train/metrics/MachRecall.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/DistributedTrainingWrapper.h>
#include <bolt/src/train/trainer/TrainState.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <optional>
#include <stdexcept>
#include <utility>

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

void defineTrainer(py::module_& train);

void defineMetrics(py::module_& train);

void defineCallbacks(py::module_& train);

void defineDistributedTrainer(py::module_& train);

void createBoltV2TrainSubmodule(py::module_& module) {
  auto train = module.def_submodule("train");

  defineTrainer(train);

#if THIRDAI_EXPOSE_ALL
  /**
   * ==============================================================
   * WARNING: If this THIRDAI_EXPOSE_ALL is removed then license
   * checks must be added to the train method.
   * ==============================================================
   */
  defineMetrics(train);
#endif

  defineCallbacks(train);

  defineDistributedTrainer(train);
}

Trainer makeTrainer(nn::model::ModelPtr model,
                    std::optional<uint32_t> freeze_hash_tables_epoch) {
  return Trainer(std::move(model), freeze_hash_tables_epoch, CtrlCCheck{});
}

void defineTrainer(py::module_& train) {
  // TODO(Nicholas): Add methods to return tensors in data pipeline and remove
  // this.

#if THIRDAI_EXPOSE_ALL
  train.def("convert_dataset", convertDataset, py::arg("dataset"),
            py::arg("dim"), py::arg("copy") = true);

  train.def("convert_datasets", convertDatasets, py::arg("datasets"),
            py::arg("dims"), py::arg("copy") = true);
#endif

  /*
   * DistributedTrainer inherits Trainer objects. Hence, we need to expose
   * constructor for Trainer class.
   */
  py::class_<Trainer>(train, "Trainer")
      .def(py::init(&makeTrainer), py::arg("model"),
           py::arg("freeze_hash_tables_epoch") = std::nullopt)
#if THIRDAI_EXPOSE_ALL
      /**
       * ==============================================================
       * WARNING: If this THIRDAI_EXPOSE_ALL is removed then license
       * checks must be added to the train method.
       * ==============================================================
       */

      .def(
          "train",
          [](Trainer& trainer, const LabeledDataset& train_data,
             float learning_rate, uint32_t epochs,
             const metrics::InputMetrics& train_metrics,
             const std::optional<LabeledDataset>& validation_data,
             const metrics::InputMetrics& validation_metrics,
             std::optional<uint32_t> steps_per_validation,
             bool use_sparsity_in_validation,
             const std::vector<callbacks::CallbackPtr>& callbacks,
             bool autotune_rehash_rebuild, bool verbose,
             std::optional<uint32_t> logging_interval, py::object& comm) {
            return trainer.train(
                train_data, learning_rate, epochs, train_metrics,
                validation_data, validation_metrics, steps_per_validation,
                use_sparsity_in_validation, callbacks, autotune_rehash_rebuild,
                verbose, logging_interval,
                (!comm.is(py::none()) ? DistributedCommPython(comm).to_optional() : std::nullopt));
          },
          py::arg("train_data"), py::arg("learning_rate"),
          py::arg("epochs") = 1,
          py::arg("train_metrics") = metrics::InputMetrics(),
          py::arg("validation_data") = std::nullopt,
          py::arg("validation_metrics") = metrics::InputMetrics(),
          py::arg("steps_per_validation") = std::nullopt,
          py::arg("use_sparsity_in_validation") = false,
          py::arg("callbacks") = std::vector<callbacks::CallbackPtr>(),
          py::arg("autotune_rehash_rebuild") = false, py::arg("verbose") = true,
          py::arg("logging_interval") = std::nullopt,
          py::arg("comm") = std::nullopt, bolt::python::OutputRedirect())
      .def(
          "train",
          [](Trainer& trainer, const LabeledDataset& train_data,
             float learning_rate, uint32_t epochs,
             std::vector<std::string>& train_metrics,
             const std::optional<LabeledDataset>& validation_data,
             std::vector<std::string>& validation_metrics,
             std::optional<uint32_t> steps_per_validation,
             bool use_sparsity_in_validation,
             const std::vector<callbacks::CallbackPtr>& callbacks,
             bool autotune_rehash_rebuild, bool verbose,
             std::optional<uint32_t> logging_interval, py::object& comm) {
            return trainer.train_with_metric_names(
                train_data, learning_rate, epochs, train_metrics,
                validation_data, validation_metrics, steps_per_validation,
                use_sparsity_in_validation, callbacks, autotune_rehash_rebuild,
                verbose, logging_interval,
              (!comm.is(py::none()) ? DistributedCommPython(comm).to_optional() : std::nullopt));
          },
          py::arg("train_data"), py::arg("learning_rate"),
          py::arg("epochs") = 1,
          py::arg("train_metrics") = std::vector<std::string>(),
          py::arg("validation_data") = std::nullopt,
          py::arg("validation_metrics") = std::vector<std::string>(),
          py::arg("steps_per_validation") = std::nullopt,
          py::arg("use_sparsity_in_validation") = false,
          py::arg("callbacks") = std::vector<callbacks::CallbackPtr>(),
          py::arg("autotune_rehash_rebuild") = false, py::arg("verbose") = true,
          py::arg("logging_interval") = std::nullopt,
          py::arg("comm") = std::nullopt, bolt::python::OutputRedirect())
      .def("validate", &Trainer::validate, py::arg("validation_data"),
           py::arg("validation_metrics") = metrics::InputMetrics(),
           py::arg("use_sparsity") = false, py::arg("verbose") = true,
           bolt::python::OutputRedirect())
      .def("validate", &Trainer::validate_with_metric_names,
           py::arg("validation_data"),
           py::arg("validation_metrics") = std::vector<std::string>(),
           py::arg("use_sparsity") = false, py::arg("verbose") = true,
           bolt::python::OutputRedirect())
      .def_property_readonly("model", &Trainer::getModel,
                             py::return_value_policy::reference_internal)
#endif
      ;
}

void defineMetrics(py::module_& train) {
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

  py::class_<metrics::FMeasure, std::shared_ptr<metrics::FMeasure>,
             metrics::Metric>(metrics, "FMeasure")
      .def(py::init<nn::autograd::ComputationPtr, nn::autograd::ComputationPtr,
                    float, float>(),
           py::arg("outputs"), py::arg("labels"), py::arg("threshold"),
           py::arg("beta") = 1);

  py::class_<metrics::MachPrecision, std::shared_ptr<metrics::MachPrecision>,
             metrics::Metric>(metrics, "MachPrecision")
      .def(py::init<dataset::mach::MachIndexPtr, uint32_t,
                    nn::autograd::ComputationPtr, nn::autograd::ComputationPtr,
                    uint32_t>(),
           py::arg("mach_index"), py::arg("top_k_per_eval_aggregation"),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));

  py::class_<metrics::MachRecall, std::shared_ptr<metrics::MachRecall>,
             metrics::Metric>(metrics, "MachRecall")
      .def(py::init<dataset::mach::MachIndexPtr, uint32_t,
                    nn::autograd::ComputationPtr, nn::autograd::ComputationPtr,
                    uint32_t>(),
           py::arg("mach_index"), py::arg("top_k_per_eval_aggregation"),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));
}

void defineCallbacks(py::module_& train) {
  auto callbacks = train.def_submodule("callbacks");

  py::class_<TrainState, TrainStatePtr>(train, "TrainState")
      .def_property("learning_rate", &TrainState::learningRate,
                    &TrainState::updateLearningRate)
      .def("stop_training", &TrainState::stopTraining);

  py::class_<callbacks::Callback, PyCallback, callbacks::CallbackPtr>(
      callbacks, "Callback")
      .def(py::init<>())
      .def_property_readonly("model", &callbacks::Callback::getModel)
      .def_property_readonly("train_state", &callbacks::Callback::getTrainState)
      .def_property_readonly("history", &callbacks::Callback::getHistory);

  py::class_<callbacks::ReduceLROnPlateau,
             std::shared_ptr<callbacks::ReduceLROnPlateau>,
             callbacks::Callback>(callbacks, "ReduceLROnPlateau")
      .def(py::init<std::string, uint32_t, uint32_t, float, float, bool, bool,
                    float>(),
           py::arg("metric"), py::arg("patience") = 10, py::arg("cooldown") = 0,
           py::arg("decay_factor") = 0.1, py::arg("threshold") = 1e-3,
           py::arg("relative_threshold") = true, py::arg("maximize") = true,
           py::arg("min_lr") = 0);

  py::class_<callbacks::Overfitting, std::shared_ptr<callbacks::Overfitting>,
             callbacks::Callback>(callbacks, "Overfitting")
      .def(py::init<std::string, float, bool>(), py::arg("metric"),
           py::arg("threshold") = 0.97, py::arg("maximize") = true);

  py::class_<callbacks::LearningRateScheduler,
             std::shared_ptr<callbacks::LearningRateScheduler>,
             callbacks::Callback>
      LearningRateScheduler(callbacks, "LearningRateScheduler");

  py::class_<callbacks::LinearSchedule,
             std::shared_ptr<callbacks::LinearSchedule>,
             callbacks::LearningRateScheduler>(callbacks, "LinearLR")
      .def(py::init<float, float, uint32_t, bool>(),
           py::arg("start_factor") = 1.0, py::arg("end_factor") = 1.0 / 3.0,
           py::arg("total_iters") = 5, py::arg("batch_level_steps") = false,
           "LinearLR scheduler changes the learning rate linearly by a small "
           "multiplicative factor until the number of epochs reaches the total "
           "iterations.\n");

  py::class_<callbacks::MultiStepLR, std::shared_ptr<callbacks::MultiStepLR>,
             callbacks::LearningRateScheduler>(callbacks, "MultiStepLR")
      .def(py::init<float, std::vector<uint32_t>, bool>(), py::arg("gamma"),
           py::arg("milestones"), py::arg("batch_level_steps") = false,
           "The Multi-step learning rate scheduler changes"
           "the learning rate by a factor of gamma for every milestone"
           "specified in the vector of milestones. \n");

  py::class_<callbacks::CosineAnnealingWarmRestart,
             std::shared_ptr<callbacks::CosineAnnealingWarmRestart>,
             callbacks::LearningRateScheduler>(callbacks,
                                               "CosineAnnealingWarmRestart")
      .def(py::init<uint32_t, uint32_t, float, bool>(),
           py::arg("initial_restart_iter") = 4,
           py::arg("iter_restart_multiplicative_factor") = 1,
           py::arg("min_lr") = 0.0, py::arg("batch_per_step") = false,
           "The cosine annealing warm restart LR scheduler decays the learning "
           "rate until the specified number of epochs (current_restart_iter) "
           "following a cosine schedule and next restarts occurs after "
           "current_restart_iter * iter_restart_multiplicative_factor");
}

void defineDistributedTrainer(py::module_& train) {
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
           py::arg("all_datasets"))
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
           &DistributedTrainingWrapper::setSerializeOptimizer,
           py::arg("should_save_optimizer"))
      .def("update_learning_rate",
           &DistributedTrainingWrapper::updateLearningRate,
           py::arg("learning_rate"))
      .def("increment_epoch_count",
           &DistributedTrainingWrapper::incrementEpochCount);
}

}  // namespace thirdai::bolt::train::python