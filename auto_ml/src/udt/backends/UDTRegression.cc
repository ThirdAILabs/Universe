#include "UDTRegression.h"
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>

namespace thirdai::automl::udt {

UDTRegression::UDTRegression(
    const data::ColumnDataTypes& input_data_types,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    const std::string& target_name, const data::NumericalDataTypePtr& target,
    std::optional<uint32_t> num_bins, std::string time_granularity,
    uint32_t lookahead, char delimiter, const config::ArgumentMap& options) {
  data::TabularBlockOptions tabular_options;

  tabular_options.contextual_columns =
      options.get<bool>("contextual_columns", "boolean", false);
  tabular_options.time_granularity = std::move(time_granularity);
  tabular_options.lookahead = lookahead;

  _binning = dataset::RegressionBinningStrategy(
      target->range.first, target->range.second, num_bins.value());

  auto label_block =
      dataset::RegressionCategoricalBlock::make(target_name, _binning, 3,
                                                /* labels_sum_to_one= */ true);

  bool force_parallel = options.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{label_block}, tabular_options, delimiter,
      force_parallel);

  uint32_t hidden_dim = options.get<uint32_t>("embedding_dim", "integer", 512);

  _model = utils::defaultModel(_dataset_factory->inputDim(), hidden_dim,
                               num_bins.value());

  _freeze_hash_tables =
      options.get<bool>("freeze_hash_tables", "boolean", true);
}

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