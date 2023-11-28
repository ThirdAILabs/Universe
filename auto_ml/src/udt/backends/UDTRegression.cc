#include "UDTRegression.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/stl.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <optional>

namespace thirdai::automl::udt {

using bolt::metrics::fromMetricNames;

UDTRegression::UDTRegression(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const NumericalDataTypePtr& target,
    std::optional<uint32_t> num_bins, const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args) {
  uint32_t output_bins = num_bins.value_or(defaults::REGRESSION_BINS);

  _model = utils::buildModel(
      /* input_dim= */ tabular_options.feature_hash_range,
      /* output_dim= */ output_bins, /* args= */ user_args,
      /* model_config= */ model_config);

  _binning = dataset::RegressionBinningStrategy(
      target->range.first, target->range.second, output_bins);

  bool normalize_target_categories = utils::hasSoftmaxOutput(_model);
  auto label_block = dataset::RegressionCategoricalBlock::make(
      target_name, _binning, defaults::REGRESSION_CORRECT_LABEL_RADIUS,
      /* labels_sum_to_one= */ normalize_target_categories);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = TabularDatasetFactory::make(
      input_data_types, temporal_tracking_relationships,
      {dataset::BlockList({label_block})}, std::set<std::string>{target_name},
      tabular_options, force_parallel);

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

py::object UDTRegression::train(const dataset::DataSourcePtr& data,
                                float learning_rate, uint32_t epochs,
                                const std::vector<std::string>& train_metrics,
                                const dataset::DataSourcePtr& val_data,
                                const std::vector<std::string>& val_metrics,
                                const std::vector<CallbackPtr>& callbacks,
                                TrainOptions options,
                                const bolt::DistributedCommPtr& comm) {
  size_t batch_size = options.batch_size.value_or(defaults::BATCH_SIZE);

  dataset::DatasetLoaderPtr val_dataset;
  if (val_data) {
    val_dataset = _dataset_factory->getLabeledDatasetLoader(
        val_data, /* shuffle= */ false);
  }

  auto train_dataset = _dataset_factory->getLabeledDatasetLoader(
      data, /* shuffle= */ true, /* shuffle_config= */ options.shuffle_config);

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});

  auto history = trainer.train_with_dataset_loader(
      /* train_data_loader= */ train_dataset,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* batch_size= */ batch_size,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */
      fromMetricNames(_model, train_metrics, /* prefix= */ "train_"),
      /* validation_data_loader= */ val_dataset,
      /* validation_metrics= */
      fromMetricNames(_model, val_metrics, /* prefix= */ "val_"),
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks,
      /* autotune_rehash_rebuild= */ true, /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /* gradient_update_interval= */ 1,
      /*comm= */ comm);

  return py::cast(history);
}

py::object UDTRegression::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference, bool verbose,
                                   std::optional<uint32_t> top_k) {
  (void)top_k;

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});

  auto dataset =
      _dataset_factory->getLabeledDatasetLoader(data, /* shuffle= */ false);

  auto history = trainer.validate_with_dataset_loader(
      dataset, fromMetricNames(_model, metrics, /* prefix= */ "val_"),
      sparse_inference, verbose);

  return py::cast(history);
}

py::object UDTRegression::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class,
                                  std::optional<uint32_t> top_k) {
  (void)return_predicted_class;  // No classes to return in regression;
  (void)top_k;

  auto output = _model->forward(_dataset_factory->featurizeInput(sample),
                                sparse_inference);

  return py::cast(unbinActivations(output.at(0)->getVector(0)));
}

py::object UDTRegression::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       std::optional<uint32_t> top_k) {
  (void)return_predicted_class;  // No classes to return in regression;
  (void)top_k;

  auto outputs = _model->forward(_dataset_factory->featurizeInputBatch(samples),
                                 sparse_inference);

  NumpyArray<float> predictions(outputs.at(0)->batchSize());
  for (uint32_t i = 0; i < outputs.at(0)->batchSize(); i++) {
    predictions.mutable_at(i) = unbinActivations(outputs.at(0)->getVector(i));
  }
  return py::object(std::move(predictions));
}

float UDTRegression::unbinActivations(const BoltVector& output) const {
  assert(output.len > 0);

  uint32_t predicted_bin_index = output.getHighestActivationId();

  return _binning.unbin(predicted_bin_index);
}

template void UDTRegression::serialize(cereal::BinaryInputArchive&,
                                       const uint32_t version);
template void UDTRegression::serialize(cereal::BinaryOutputArchive&,
                                       const uint32_t version);

template <class Archive>
void UDTRegression::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_REGRESSION";
  versions::checkVersion(version, versions::UDT_REGRESSION_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_REGRESSION_VERSION after serialization
  // changes
  archive(cereal::base_class<UDTBackend>(this), _model, _dataset_factory,
          _binning, _freeze_hash_tables);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTRegression)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTRegression,
                     thirdai::versions::UDT_REGRESSION_VERSION)