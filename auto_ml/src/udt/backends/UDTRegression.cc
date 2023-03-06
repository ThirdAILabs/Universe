#include "UDTRegression.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <pybind11/stl.h>

namespace thirdai::automl::udt {

UDTRegression::UDTRegression(const data::ColumnDataTypes& input_data_types,
                             const data::UserProvidedTemporalRelationships&
                                 temporal_tracking_relationships,
                             const std::string& target_name,
                             const data::NumericalDataTypePtr& target,
                             std::optional<uint32_t> num_bins,
                             const data::TabularOptions& tabular_options,
                             const std::optional<std::string>& model_config,
                             const config::ArgumentMap& user_args) {
  uint32_t output_bins = num_bins.value_or(defaults::REGRESSION_BINS);

  if (model_config) {
    _model = utils::loadModel({tabular_options.feature_hash_range}, output_bins,
                              *model_config);
  } else {
    uint32_t hidden_dim = user_args.get<uint32_t>(
        "embedding_dimension", "integer", defaults::HIDDEN_DIM);
    _model = utils::defaultModel(tabular_options.feature_hash_range, hidden_dim,
                                 output_bins);
  }

  _binning = dataset::RegressionBinningStrategy(
      target->range.first, target->range.second, output_bins);

  bool normalize_target_categories = utils::hasSoftmaxOutput(_model);
  auto label_block = dataset::RegressionCategoricalBlock::make(
      target_name, _binning, defaults::REGRESSION_CORRECT_LABEL_RADIUS,
      /* labels_sum_to_one= */ normalize_target_categories);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{label_block},
      std::set<std::string>{target_name}, tabular_options, force_parallel, label_block->delimiter(), label_block->columnName());

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

void UDTRegression::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, metrics, callbacks, verbose,
      logging_interval, _dataset_factory);

  auto train_dataset =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  utils::train(_model, train_dataset, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables);
}

py::object UDTRegression::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference,
                                   bool return_predicted_class, bool verbose,
                                   bool return_metrics) {
  (void)return_predicted_class;  // No classes to return in regression;

  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ false)
          ->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);

  auto [output_metrics, output] =
      _model->evaluate(test_data, test_labels, eval_config);
  if (return_metrics) {
    return py::cast(output_metrics);
  }

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

  utils::NumpyArray<float> predictions(outputs.getBatchSize());
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

template void UDTRegression::serialize(cereal::BinaryInputArchive&);
template void UDTRegression::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTRegression::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _model, _dataset_factory,
          _binning, _freeze_hash_tables);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTRegression)