#include "BoltTrainPython.h"
#include "CtrlCCheck.h"
#include "DistributedCommunicationPython.h"
#include "PyCallback.h"
#include "PybindUtils.h"
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
#include <bolt/src/train/trainer/TrainState.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace thirdai::bolt::python {

void defineTrainer(py::module_& train);

void defineMetrics(py::module_& train);

void defineCallbacks(py::module_& train);

void defineDistributedTrainer(py::module_& train);

void createBoltTrainSubmodule(py::module_& module) {
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

Trainer makeTrainer(ModelPtr model,
                    std::optional<uint32_t> freeze_hash_tables_epoch,
                    uint32_t gradient_update_interval) {
  return Trainer(std::move(model), freeze_hash_tables_epoch,
                 gradient_update_interval, CtrlCCheck{});
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
           py::arg("freeze_hash_tables_epoch") = std::nullopt,
           py::arg("gradient_update_interval") = 1)
#if THIRDAI_EXPOSE_ALL
      /**
       * ==============================================================
       * WARNING: If this THIRDAI_EXPOSE_ALL is removed then license
       * checks must be added to the train method.
       * ==============================================================
       */

      .def("train", &Trainer::train, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs") = 1,
           py::arg("train_metrics") = metrics::InputMetrics(),
           py::arg("validation_data") = std::nullopt,
           py::arg("validation_metrics") = metrics::InputMetrics(),
           py::arg("steps_per_validation") = std::nullopt,
           py::arg("use_sparsity_in_validation") = false,
           py::arg("callbacks") = std::vector<callbacks::CallbackPtr>(),
           py::arg("autotune_rehash_rebuild") = false,
           py::arg("verbose") = true,
           py::arg("logging_interval") = std::nullopt,
           py::arg("comm") = nullptr, bolt::python::OutputRedirect())
      .def("train", &Trainer::train_with_metric_names, py::arg("train_data"),
           py::arg("learning_rate"), py::arg("epochs") = 1,
           py::arg("train_metrics") = std::vector<std::string>(),
           py::arg("validation_data") = std::nullopt,
           py::arg("validation_metrics") = std::vector<std::string>(),
           py::arg("steps_per_validation") = std::nullopt,
           py::arg("use_sparsity_in_validation") = false,
           py::arg("callbacks") = std::vector<callbacks::CallbackPtr>(),
           py::arg("autotune_rehash_rebuild") = false,
           py::arg("verbose") = true,
           py::arg("logging_interval") = std::nullopt,
           py::arg("comm") = nullptr, bolt::python::OutputRedirect())
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
      .def(py::init<LossPtr>(), py::arg("loss_fn"));

  py::class_<metrics::CategoricalAccuracy,
             std::shared_ptr<metrics::CategoricalAccuracy>, metrics::Metric>(
      metrics, "CategoricalAccuracy")
      .def(py::init<ComputationPtr, ComputationPtr>(), py::arg("outputs"),
           py::arg("labels"));

  py::class_<metrics::PrecisionAtK, std::shared_ptr<metrics::PrecisionAtK>,
             metrics::Metric>(metrics, "PrecisionAtK")
      .def(py::init<ComputationPtr, ComputationPtr, uint32_t>(),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));

  py::class_<metrics::RecallAtK, std::shared_ptr<metrics::RecallAtK>,
             metrics::Metric>(metrics, "RecallAtK")
      .def(py::init<ComputationPtr, ComputationPtr, uint32_t>(),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));

  py::class_<metrics::FMeasure, std::shared_ptr<metrics::FMeasure>,
             metrics::Metric>(metrics, "FMeasure")
      .def(py::init<ComputationPtr, ComputationPtr, float, float>(),
           py::arg("outputs"), py::arg("labels"), py::arg("threshold"),
           py::arg("beta") = 1);

  py::class_<metrics::MachPrecision, std::shared_ptr<metrics::MachPrecision>,
             metrics::Metric>(metrics, "MachPrecision")
      .def(py::init<dataset::mach::MachIndexPtr, uint32_t, ComputationPtr,
                    ComputationPtr, uint32_t>(),
           py::arg("mach_index"), py::arg("num_buckets_to_eval"),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));

  py::class_<metrics::MachRecall, std::shared_ptr<metrics::MachRecall>,
             metrics::Metric>(metrics, "MachRecall")
      .def(py::init<dataset::mach::MachIndexPtr, uint32_t, ComputationPtr,
                    ComputationPtr, uint32_t>(),
           py::arg("mach_index"), py::arg("num_buckets_to_eval"),
           py::arg("outputs"), py::arg("labels"), py::arg("k"));
}

void defineCallbacks(py::module_& train) {
  auto callbacks = train.def_submodule("callbacks");

  py::class_<TrainState, TrainStatePtr>(train, "TrainState")
      .def_property("learning_rate", &TrainState::learningRate,
                    &TrainState::updateLearningRate)
      .def("stop_training", &TrainState::stopTraining)
      .def("batches_in_dataset", &TrainState::batchesInDataset);

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
      .def(py::init<float, float, uint32_t, uint32_t, uint32_t, bool>(),
           py::arg("min_lr"), py::arg("max_lr"), py::arg("steps_until_restart"),
           py::arg("linear_warmup_steps") = 0,
           py::arg("steps_until_restart_scaling_factor") = 1,
           py::arg("batch_level_steps") = true,
           "The cosine annealing warm restart LR scheduler decays the learning "
           "rate until the specified number of steps (steps_until_restart) "
           "following a cosine schedule and the next restart occurs after "
           "steps_until_restart * steps_until_restart_scaling_factor");
}

void defineDistributedTrainer(py::module_& train) {
  py::class_<DistributedComm, PyDistributedComm, DistributedCommPtr>(
      train, "Communication")
      .def(py::init<>());
}

}  // namespace thirdai::bolt::python