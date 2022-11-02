#include "CallbacksPython.h"
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/callbacks/EarlyStopCheckpoint.h>
#include <bolt/src/callbacks/LearningRateScheduler.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createCallbacksSubmodule(py::module_& module) {
  auto callbacks_submodule = module.def_submodule("callbacks");

  py::class_<Callback, PyCallback, CallbackPtr>(callbacks_submodule, "Callback")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<>())
      .def("on_train_begin", &Callback::onTrainBegin)
      .def("on_train_end", &Callback::onTrainEnd)
      .def("on_epoch_begin", &Callback::onEpochBegin)
      .def("on_epoch_end", &Callback::onEpochEnd)
      .def("on_batch_begin", &Callback::onBatchBegin)
      .def("on_batch_end", &Callback::onBatchEnd);

  py::class_<TrainState>(callbacks_submodule, "TrainState")
      .def_readwrite("learning_rate", &TrainState::learning_rate)
      .def_readwrite("verbose", &TrainState::verbose)
      .def_readwrite("rebuild_hash_tables_batch",
                     &TrainState::rebuild_hash_tables_batch)
      .def_readwrite("reconstruct_hash_functions_batch",
                     &TrainState::reconstruct_hash_functions_batch)
      .def_readwrite("stop_training", &TrainState::stop_training)
      .def_readonly("epoch_times", &TrainState::epoch_times)
      .def("get_train_metrics", &TrainState::getTrainMetrics,
           py::arg("metric_name"))
      .def("get_all_train_metrics", &TrainState::getAllTrainMetrics)
      .def("get_validation_metrics", &TrainState::getValidationMetrics,
           py::arg("metric_name"))
      .def("get_all_validation_metrics", &TrainState::getAllValidationMetrics);
#endif

  py::class_<LRSchedule, LRSchedulePtr>(callbacks_submodule,  // NOLINT
                                        "LRSchedule");        // NOLINT

  py::class_<MultiplicativeLR, MultiplicativeLRPtr, LRSchedule>(
      callbacks_submodule, "MultiplicativeLR")
      .def(py::init<float>(), py::arg("gamma"),
           "The Multiplicative learning rate scheduler "
           "multiplies the current learning rate by gamma every epoch.\n");

  py::class_<ExponentialLR, ExponentialLRPtr, LRSchedule>(callbacks_submodule,
                                                          "ExponentialLR")
      .def(py::init<float>(), py::arg("gamma"),
           "The exponential learning rate scheduler decays the learning"
           "rate by an exponential factor of gamma for every epoch.\n");

  py::class_<MultiStepLR, MultiStepLRPtr, LRSchedule>(callbacks_submodule,
                                                      "MultiStepLR")
      .def(py::init<float, std::vector<uint32_t>>(), py::arg("gamma"),
           py::arg("milestones"),
           "The Multi-step learning rate scheduler changes"
           "the learning rate by a factor of gamma for every milestone"
           "specified in the vector of milestones. \n");

  py::class_<LambdaSchedule, LambdaSchedulePtr, LRSchedule>(callbacks_submodule,
                                                            "LambdaSchedule")
      .def(py::init<const std::function<float(float, uint32_t)>&>(),
           py::arg("schedule"),
           "The Lambda scheduler changes the learning rate depending "
           "on a custom lambda function."
           "Arguments:\n"
           " * schedule: learning rate schedule function with signature \n"
           "         float schedule(float learning_rate, uint32_t epoch)\n");

  py::class_<LearningRateScheduler, LearningRateSchedulerPtr, Callback>(
      callbacks_submodule, "LearningRateScheduler")
      .def(py::init<LRSchedulePtr>(), py::arg("schedule"))
      .def("get_final_lr", &LearningRateScheduler::getFinalLR);

  py::class_<KeyboardInterrupt, KeyboardInterruptPtr, Callback>(
      callbacks_submodule, "KeyboardInterrupt")
      .def(py::init<>());

  py::class_<EarlyStopCheckpoint, EarlyStopCheckpointPtr, Callback>(
      callbacks_submodule, "EarlyStopCheckpoint")
      .def(
          py::init<std::string, std::string, uint32_t, double>(),
          py::arg("monitored_metric"), py::arg("model_save_path"),
          py::arg("patience"), py::arg("min_delta"),
          "This callback is intended to stop training early based on prediction"
          " results from a given validation set. Saves the best model to "
          "model_save_path.\n"
          "Arguments:\n"
          " * monitored_metric: The metric to monitor for early stopping. The "
          "metric is assumed to be associated with validation data.\n"
          " * model_save_path: string. The file path to save the model that "
          "scored the best on the validation set\n"
          " * patience: int. The nuber of epochs with no improvement in "
          "validation score after which training will be stopped.\n"
          " * min_delta: float. The minimum change in the monitored quantity "
          "to qualify as an improvement, i.e. an absolute change of less than "
          "min_delta will count as no improvement.\n");
}

}  // namespace thirdai::bolt::python