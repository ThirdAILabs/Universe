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

  py::class_<Callback, PyCallback, CallbackPtr> py_callback(callbacks_submodule,
                                                            "Callback");
  // We don't expose custom callbacks to external users of our python package
  // because we don't want them messing with the internal BoltGraph. We also
  // consequently hide TrainState since it's only used in callbacks.
#if THIRDAI_EXPOSE_ALL
  py_callback.def(py::init<>())
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
#else
  (void)py_callback;
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

#if THIRDAI_EXPOSE_ALL
  py::class_<KeyboardInterrupt, KeyboardInterruptPtr, Callback>(
      callbacks_submodule, "KeyboardInterrupt")
      .def(py::init<>());
#endif

  py::class_<EarlyStopCheckpoint, EarlyStopCheckpointPtr, Callback>(
      callbacks_submodule, "EarlyStopCheckpoint")
      .def(py::init<std::string, std::optional<std::string>, uint32_t, uint32_t,
                    float, double, std::string, std::optional<double>>(),
           py::arg("model_save_path"),
           py::arg("monitored_metric") = std::nullopt, py::arg("patience") = 2,
           py::arg("max_lr_adjustments") = 2, py::arg("lr_multiplier") = 0.5,
           py::arg("min_delta") = 0, py::arg("compare_against") = "prev",
           py::arg("time_out") = std::nullopt, R"pbdoc(
This callback monitors a validation metric and gives users a means to configure 
their model training based on that metric. It provides features for saving 
the best scoring model on the validation set, stopping training early when the model converges, adjusting
the learning rate, and adding a training timeout.
Args:
     model_save_path (string): The file path to save the model that scored the 
          best on the validation set.
     monitored_metric (string): Optional: The metric to monitor for early stopping.
     If there is no metric specified we will use the validation metric provided. 
     We will throw an error if there are no tracked validation metrics, if 
     validation is not set up, or if there are multiple validation metrics.
     patience (int): The number of epochs with no improvement in validation score
          after which we will evaluate whether to do one of two things: 1) adjust
          the learning rate and continue training or 2) stop training if we've 
          changed the learning rate enough times. Defaults to 2.
     max_lr_adjustments (int): The maximum number of learning rate adjustments 
          allowed after a "patience" interval. Defaults to 2.
     lr_multiplier (float): Multiplier for the learning rate after a 'patience' 
          interval. Defaults to 0.5. Must be positive.
     min_delta (float): The minimum change in the monitored metric to qualify 
          as an improvement, i.e. an absolute change of less than min_delta will
          count as no improvement. Defaults to 0. 
     compare_against (string): One of 'best' or 'prev'. Determines whether to 
          compare against the best validation metric so far or the previous validation
          metric recorded. Defaults to 'prev'.
     time_out (float): Optional. Represents the total training time (in seconds)
     after which the model will stop training. Rounds up to the nearest epoch.
)pbdoc");
}

}  // namespace thirdai::bolt::python