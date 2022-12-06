#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/Categorical.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;

namespace thirdai::automl::models {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

/**
 * This class is an interface to abstract converting the output of the model in
 * a ModelPipeline to a format that is convenient for the user.
 */
class OutputProcessor {
 public:
  // Processes output from predict. The return_predicted_class option indicates
  // that it should return the predicted class rather than the activations, this
  // has no effect on regression outputs.
  virtual py::object processBoltVector(BoltVector& output,
                                       bool return_predicted_class) = 0;

  // Processes output from predictBatch. The return_predicted_class option
  // indicates that it should return the predicted class rather than the
  // activations, this has no effect on regression outputs.
  virtual py::object processBoltBatch(BoltBatch& outputs,
                                      bool return_predicted_class) = 0;

  // Processes output from evaluate. The return_predicted_class option indicates
  // that it should return the predicted class rather than the activations, this
  // has no effect on regression outputs.
  virtual py::object processOutputTracker(bolt::InferenceOutputTracker& output,
                                          bool return_predicted_class) = 0;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using OutputProcessorPtr = std::shared_ptr<OutputProcessor>;

/**
 * This class performs output processing for classification problems. It simply
 * returns numpy arrays representing the output. If the output is sparse then
 * two numpy arrays are returned, one for the indices and one for the values. If
 * the output is dense the only one array is returned.
 *
 * The prediction_threshold parameter is optional and is intended for use with
 * multi-class classification where the predicted classes are determined by all
 * the neurons with activations exceeding some threshold. When specified it
 * ensures that the largest activation is always >= the threshold, which means
 * that at least one class will be predicted for every input. This can boost
 * accuracy on some datasets.
 */
class CategoricalOutputProcessor final : public OutputProcessor {
 public:
  explicit CategoricalOutputProcessor(std::optional<float> prediction_threshold)
      : _prediction_threshold(prediction_threshold) {}

  static auto make(std::optional<float> prediction_threshold = std::nullopt) {
    return std::make_shared<CategoricalOutputProcessor>(prediction_threshold);
  }

  py::object processBoltVector(BoltVector& output,
                               bool return_predicted_class) final;

  py::object processBoltBatch(BoltBatch& outputs,
                              bool return_predicted_class) final;

  py::object processOutputTracker(bolt::InferenceOutputTracker& output,
                                  bool return_predicted_class) final;

 private:
  void ensureMaxActivationLargerThanThreshold(float* activations, uint32_t len);

  std::optional<float> _prediction_threshold;

  // Private constructor for cereal.
  CategoricalOutputProcessor() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OutputProcessor>(this), _prediction_threshold);
  }
};

/**
 * This class performs output processing for regression problems. It takes the
 * categorical output from the model and maps the predicted bins back to
 * continuous values that users will expect from a regression model.
 */
class RegressionOutputProcessor final : public OutputProcessor {
 public:
  explicit RegressionOutputProcessor(
      dataset::RegressionBinningStrategy regression_binning)
      : _regression_binning(regression_binning) {}

  static auto make(dataset::RegressionBinningStrategy regression_binning) {
    return std::make_shared<RegressionOutputProcessor>(regression_binning);
  }

  py::object processBoltVector(BoltVector& output,
                               bool return_predicted_class) final;

  py::object processBoltBatch(BoltBatch& outputs,
                              bool return_predicted_class) final;

  py::object processOutputTracker(bolt::InferenceOutputTracker& output,
                                  bool return_predicted_class) final;

 private:
  float unbinActivations(const BoltVector& output) const;

  dataset::RegressionBinningStrategy _regression_binning;

  // Private constructor for cereal.
  RegressionOutputProcessor() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OutputProcessor>(this), _regression_binning);
  }
};

/**
 * This class performs output processing for binary classification problems.
 * This is different from the regular CategoricalOutputProcessor because it
 * allows for a custom threshold to be set for positive predictions when the
 * return_predicted_class option is used. For example if the output
 * probabilities are [0.8, 0.2] but the threshold is set to 0.1, then it will
 * output a prediction of 1 instead of 0.
 */
class BinaryOutputProcessor final : public OutputProcessor {
 public:
  BinaryOutputProcessor() {}

  static auto make() { return std::make_shared<BinaryOutputProcessor>(); }

  static auto cast(const OutputProcessorPtr& output_processor) {
    return std::dynamic_pointer_cast<BinaryOutputProcessor>(output_processor);
  }

  py::object processBoltVector(BoltVector& output,
                               bool return_predicted_class) final;

  py::object processBoltBatch(BoltBatch& outputs,
                              bool return_predicted_class) final;

  py::object processOutputTracker(bolt::InferenceOutputTracker& output,
                                  bool return_predicted_class) final;

  /**
   * Sets the prediction threshold for class 1. This is used by the
   * ModelPipeline after selecting the threshold which maximizies the metric of
   * interest.
   */
  void setPredictionTheshold(std::optional<float> threshold) {
    _prediction_threshold = threshold;
  }

 private:
  uint32_t binaryActivationsToPrediction(const float* activations);

  std::optional<float> _prediction_threshold;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OutputProcessor>(this), _prediction_threshold);
  }
};

using BinaryOutputProcessorPtr = std::shared_ptr<BinaryOutputProcessor>;

// Helper function for InferenceOutputTracker to Numpy.
py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

// Helper function for BoltVector to Numpy.
py::object convertBoltVectorToNumpy(const BoltVector& vector);

// Helper function for BoltBatch to Numpy.
py::object convertBoltBatchToNumpy(const BoltBatch& batch);

// Helper function used for OutputProcessors.
uint32_t argmax(const float* array, uint32_t len);

}  // namespace thirdai::automl::models