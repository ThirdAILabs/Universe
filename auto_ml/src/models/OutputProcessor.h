#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/Categorical.h>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;

namespace thirdai::automl::models {

/**
 * This class is an interface to abstract converting the output of the model in
 * a ModelPipeline to a format that is convenient for the user.
 */
class OutputProcessor {
 public:
  // Processes output from predict.
  virtual py::object processBoltVector(BoltVector& output) = 0;

  // Processes output from predictBatch.
  virtual py::object processBoltBatch(BoltBatch& outputs) = 0;

  // Processes output from evaluate.
  virtual py::object processOutputTracker(
      bolt::InferenceOutputTracker& output) = 0;

  virtual ~OutputProcessor() = default;

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

  py::object processBoltVector(BoltVector& output) final;

  py::object processBoltBatch(BoltBatch& outputs) final;

  py::object processOutputTracker(bolt::InferenceOutputTracker& output) final;

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

  py::object processBoltVector(BoltVector& output) final;

  py::object processBoltBatch(BoltBatch& outputs) final;

  py::object processOutputTracker(bolt::InferenceOutputTracker& output) final;

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

// Helper function for InferenceOutputTracker to Numpy.
py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

// Helper function for BoltVector to Numpy.
py::object convertBoltVectorToNumpy(const BoltVector& vector);

// Helper function for BoltBatch to Numpy.
py::object convertBoltBatchToNumpy(const BoltBatch& batch);

// Helper function used for OutputProcessors.
uint32_t argmax(const float* array, uint32_t len);

}  // namespace thirdai::automl::models