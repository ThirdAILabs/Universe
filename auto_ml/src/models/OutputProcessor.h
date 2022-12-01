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

class OutputProcessor {
 public:
  virtual py::object processBoltVector(BoltVector& output) = 0;

  virtual py::object processBoltBatch(BoltBatch& outputs) = 0;

  virtual py::object processOutputTracker(
      bolt::InferenceOutputTracker& output) = 0;

  static py::object convertInferenceTrackerToNumpy(
      bolt::InferenceOutputTracker& output);

  static py::object convertBoltVectorToNumpy(const BoltVector& vector);

  static py::object convertBoltBatchToNumpy(const BoltBatch& batch);

  static uint32_t argmax(const float* const array, uint32_t len) {
    assert(len > 0);

    uint32_t max_index = 0;
    float max_value = array[0];
    for (uint32_t i = 1; i < len; i++) {
      if (array[i] > max_value) {
        max_index = i;
        max_value = array[i];
      }
    }
    return max_index;
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using OutputProcessorPtr = std::shared_ptr<OutputProcessor>;

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
  std::optional<float> _prediction_threshold;

  // Private constructor for cereal.
  CategoricalOutputProcessor() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OutputProcessor>(this), _prediction_threshold);
  }
};

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
  dataset::RegressionBinningStrategy _regression_binning;

  // Private constructor for cereal.
  RegressionOutputProcessor() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OutputProcessor>(this), _regression_binning);
  }
};

class BinaryOutputProcessor final : public OutputProcessor {
 public:
  BinaryOutputProcessor() {}

  static auto make() { return std::make_shared<BinaryOutputProcessor>(); }

  static auto cast(const OutputProcessorPtr& output_processor) {
    return std::dynamic_pointer_cast<BinaryOutputProcessor>(output_processor);
  }

  py::object processBoltVector(BoltVector& output) final;

  py::object processBoltBatch(BoltBatch& outputs) final;

  py::object processOutputTracker(bolt::InferenceOutputTracker& output) final;

  void setPredictionTheshold(std::optional<float> threshold) {
    _prediction_threshold = threshold;
  }

 private:
  std::optional<float> _prediction_threshold;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OutputProcessor>(this), _prediction_threshold);
  }
};

using BinaryOutputProcessorPtr = std::shared_ptr<BinaryOutputProcessor>;

}  // namespace thirdai::automl::models