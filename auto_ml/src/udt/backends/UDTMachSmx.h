#pragma once

#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/MachModel.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/SmxTensorConversion.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/DataSource.h>
#include <smx/src/autograd/functions/Loss.h>
#include <smx/src/optimizers/Adam.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::automl::udt {

class UDTMachSmx final : public UDTBackend {
 public:
  UDTMachSmx(
      ColumnDataTypes input_data_types,
      const UserProvidedTemporalRelationships& temporal_tracking_relationships,
      const std::string& target_name, const CategoricalDataTypePtr& target,
      uint32_t n_target_classes, bool integer_target,
      const TabularOptions& tabular_options,
      const std::optional<std::string>& model_config,
      const config::ArgumentMap& args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options,
                   const bolt::DistributedCommPtr& comm) final;

  py::object coldstart(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      float learning_rate, uint32_t epochs,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics,
      const std::vector<CallbackPtr>& callbacks, TrainOptions options,
      const bolt::DistributedCommPtr& comm) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final;

  //   void introduceDocuments(const dataset::DataSourcePtr& data,
  //                           const std::vector<std::string>&
  //                           strong_column_names, const
  //                           std::vector<std::string>& weak_column_names,
  //                           std::optional<uint32_t> num_buckets_to_sample,
  //                           uint32_t num_random_hashes, bool
  //                           fast_approximation, bool verbose, bool
  //                           sort_random_hashes) final;

 private:
  data::TransformationPtr coldStartTransform(
      const std::vector<std::string>& strong_cols,
      const std::vector<std::string>& weak_cols,
      std::optional<data::VariableLengthConfig> variable_length,
      bool fast_approximation) const;

  using TrainingDataset =
      std::vector<std::pair<smx::VariablePtr, smx::VariablePtr>>;

  TrainingDataset loadTrainingData(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig(),
      const data::TransformationPtr& cold_start = nullptr) const;

  using EvalDataset = std::vector<
      std::pair<smx::VariablePtr, std::vector<std::vector<uint32_t>>>>;

  EvalDataset loadEvalData(
      const dataset::DataSourcePtr& data_source, size_t batch_size,
      const data::TransformationPtr& cold_start = nullptr) const;

  smx::VariablePtr featurizeInput(data::ColumnMap&& columns) {
    columns = _text_transform->applyStateless(std::move(columns));
    auto tensors = data::toSmxTensors(columns, _input_columns);
    return smx::Variable::make(tensors.at(0), false);
  }

  void train(const TrainingDataset& dataset, float learning_rate);

  std::vector<std::pair<uint32_t, double>> decode(float* scores,
                                                  uint32_t top_k) {
    BoltVector vec(/*an=*/nullptr, /*a=*/scores, /*g=*/nullptr,
                   _mach_index->numBuckets());
    return _mach_index->decode(vec, top_k, _default_num_buckets_to_eval);
  }

  smx::DenseTensorPtr getScores(const smx::TensorPtr& out) const {
    if (_softmax) {
      return smx::dense(smx::softmax(out));
    }
    return smx::dense(smx::sigmoid(out));
  }

  data::OutputColumnsList _input_columns;
  data::OutputColumnsList _label_columns;

  std::shared_ptr<data::TextTokenizer> _text_transform;
  data::TransformationPtr _entity_parse_transform;
  std::shared_ptr<data::MachLabel> _mach_label_transform;
  data::MachIndexPtr _mach_index;
  char _delimiter;

  std::unique_ptr<MachModel> _model;
  std::unique_ptr<smx::Adam> _optimizer;
  bool _softmax;

  bool _freeze_hash_tables;
  uint32_t _default_topk = defaults::MACH_TOP_K_TO_RETURN;
  uint32_t _default_num_buckets_to_eval = defaults::MACH_NUM_BUCKETS_TO_EVAL;

  uint32_t _epoch = 0;
};

}  // namespace thirdai::automl::udt