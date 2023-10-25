#include "UDTRegression.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/Loader.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/RegressionBinning.h>
#include <data/src/transformations/StringCast.h>
#include <pybind11/stl.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <memory>
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

  auto cast = std::make_shared<data::StringToDecimal>(target_name, target_name);

  _binning = std::make_shared<data::RegressionBinning>(
      target_name, FEATURIZED_LABELS, target->range.first, target->range.second,
      output_bins, defaults::REGRESSION_CORRECT_LABEL_RADIUS);

  auto label_transform = data::Pipeline::make({cast, _binning});

  bool softmax_output = utils::hasSoftmaxOutput(_model);
  data::ValueFillType value_fill = softmax_output
                                       ? data::ValueFillType::SumToOne
                                       : data::ValueFillType::Ones;

  data::OutputColumnsList bolt_labels = {
      data::OutputColumns(FEATURIZED_LABELS, value_fill)};

  auto temporal_relationships = TemporalRelationshipsAutotuner::autotune(
      input_data_types, temporal_tracking_relationships,
      tabular_options.lookahead);

  _featurizer = std::make_shared<Featurizer>(
      input_data_types, temporal_relationships, target_name, label_transform,
      bolt_labels, tabular_options);
}

UDTRegression::UDTRegression(const proto::udt::UDTRegression& regression,
                             bolt::ModelPtr model)
    : _model(std::move(model)),
      _featurizer(std::make_shared<Featurizer>(regression.featurizer())) {
  auto binning = thirdai::data::Transformation::fromProto(regression.binning());

  // The toProto method on the regression binning transformation returns a
  // Transformation proto object, and when we invoke Transformation::fromProto
  // we get an instance of the Transformation base class, thus we need to
  // downcast it to get the regression binning transformation.
  _binning =
      std::dynamic_pointer_cast<thirdai::data::RegressionBinning>(binning);
  if (!_binning) {
    throw std::invalid_argument(
        "Expected regression label transformation in fromProto.");
  }
}

py::object UDTRegression::train(const dataset::DataSourcePtr& data,
                                float learning_rate, uint32_t epochs,
                                const std::vector<std::string>& train_metrics,
                                const dataset::DataSourcePtr& val_data,
                                const std::vector<std::string>& val_metrics,
                                const std::vector<CallbackPtr>& callbacks,
                                TrainOptions options,
                                const bolt::DistributedCommPtr& comm) {
  auto train_data_loader =
      _featurizer->getDataLoader(data, options.batchSize(), /* shuffle= */ true,
                                 options.verbose, options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader = _featurizer->getDataLoader(
        val_data, defaults::BATCH_SIZE, /* shuffle= */ false, options.verbose);
  }

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});

  auto history = trainer.train_with_data_loader(
      /* train_data_loader= */ train_data_loader,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */
      fromMetricNames(_model, train_metrics, /* prefix= */ "train_"),
      /* validation_data_loader= */ val_data_loader,
      /* validation_metrics= */
      fromMetricNames(_model, val_metrics, /* prefix= */ "val_"),
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks,
      /* autotune_rehash_rebuild= */ true, /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);

  return py::cast(history);
}

py::object UDTRegression::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference, bool verbose,
                                   std::optional<uint32_t> top_k) {
  (void)top_k;

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});

  auto data_loader = _featurizer->getDataLoader(data, defaults::BATCH_SIZE,
                                                /* shuffle= */ false, verbose);

  auto history = trainer.validate_with_data_loader(
      data_loader, fromMetricNames(_model, metrics, /* prefix= */ "val_"),
      sparse_inference, verbose);

  return py::cast(history);
}

py::object UDTRegression::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class,
                                  std::optional<uint32_t> top_k) {
  (void)return_predicted_class;  // No classes to return in regression;
  (void)top_k;

  auto output =
      _model->forward(_featurizer->featurizeInput(sample), sparse_inference);

  return py::cast(unbinActivations(output.at(0)->getVector(0)));
}

py::object UDTRegression::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       std::optional<uint32_t> top_k) {
  (void)return_predicted_class;  // No classes to return in regression;
  (void)top_k;

  auto outputs = _model->forward(_featurizer->featurizeInputBatch(samples),
                                 sparse_inference);

  NumpyArray<float> predictions(outputs.at(0)->batchSize());
  for (uint32_t i = 0; i < outputs.at(0)->batchSize(); i++) {
    predictions.mutable_at(i) = unbinActivations(outputs.at(0)->getVector(i));
  }
  return py::object(std::move(predictions));
}

proto::udt::UDT UDTRegression::toProto() const {
  proto::udt::UDT udt;

  auto* regression = udt.mutable_regression();

  regression->set_allocated_featurizer(_featurizer->toProto());
  regression->set_allocated_binning(_binning->toProto());

  return udt;
}

float UDTRegression::unbinActivations(const BoltVector& output) const {
  assert(output.len > 0);

  uint32_t predicted_bin_index = output.getHighestActivationId();

  return _binning->unbin(predicted_bin_index);
}

}  // namespace thirdai::automl::udt
