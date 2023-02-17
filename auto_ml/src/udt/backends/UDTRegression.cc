#include "UDTRegression.h"
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Train.h>

namespace thirdai::automl::udt {

void UDTRegression::train(
    const dataset::DataSourcePtr& train_data, uint32_t epochs,
    float learning_rate, const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  size_t batch_size = batch_size_opt.value_or(utils::DEFAULT_BATCH_SIZE);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, train_metrics, callbacks, verbose,
      logging_interval, _dataset_factory);

  auto train_dataset =
      _dataset_factory->getDatasetLoader(train_data, /* training= */ true);

  utils::train(_model, train_dataset, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables);
}

py::object UDTRegression::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference,
                                   bool return_predicted_class, bool verbose) {
  (void)return_predicted_class;  // No classes to return in regression;

  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      _dataset_factory->getDatasetLoader(data, /* training= */ false)
          ->loadAll(/* batch_size= */ utils::DEFAULT_BATCH_SIZE, verbose);

  auto [_, output] = _model->evaluate(test_data, test_labels, eval_config);

  utils::NumpyArray<float> output_array(output.numSamples());

  for (uint32_t i = 0; i < output.numSamples(); i++) {
    BoltVector ith_sample = output.getSampleAsNonOwningBoltVector(i);
    output_array.mutable_at(i) = unbinActivations(ith_sample);
  }

  return py::object(std::move(output_array));
}

py::object UDTRegression::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class) {
  (void)return_predicted_class;  // No classes to return in regression;

  BoltVector output = _model->predictSingle(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  return py::cast(unbinActivations(output));
}

py::object UDTRegression::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class) {
  (void)return_predicted_class;  // No classes to return in regression;

  BoltBatch outputs = _model->predictSingleBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference);

  utils::NumpyArray<uint32_t> predictions(outputs.getBatchSize());
  for (uint32_t i = 0; i < outputs.getBatchSize(); i++) {
    predictions.mutable_at(i) = unbinActivations(outputs[i]);
  }
  return py::object(std::move(predictions));
}

float UDTRegression::unbinActivations(const BoltVector& output) const {
  assert(output.len > 0);

  uint32_t predicted_bin_index = output.getHighestActivationId();

  return _binning.unbin(predicted_bin_index);
}

}  // namespace thirdai::automl::udt