#pragma once

#include <auto_ml/src/featurization/ReservedColumns.h>
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
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/DataSource.h>
#include <smx/src/autograd/functions/Loss.h>
#include <smx/src/optimizers/Adam.h>

namespace thirdai::automl::udt {

class UDTMachSmx final : public UDTBackend {
 public:
  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options,
                   const bolt::DistributedCommPtr& comm) final {
    CHECK(train_metrics.empty(), "Arg 'train_metrics' not supported.");
    CHECK(!val_data, "Arg 'val_data' not supported");
    CHECK(val_metrics.empty(), "Arg 'val_metrics' not supported");
    CHECK(callbacks.empty(), "Arg 'callbacks' not supported");
    CHECK(!options.max_in_memory_batches, "Arg 'max_in_memory' not supported");
    CHECK(!comm, "Arg 'comm' not supported");

    auto dataset = loadData(data, options.batchSize(), /*shuffle*/ true,
                            options.shuffle_config);

    for (uint32_t e = 0; e < epochs; e++) {
      train(dataset, learning_rate);
    }

    return py::none();
  }

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
      const bolt::DistributedCommPtr& comm) final {
    CHECK(train_metrics.empty(), "Arg 'train_metrics' not supported.");
    CHECK(!val_data, "Arg 'val_data' not supported");
    CHECK(val_metrics.empty(), "Arg 'val_metrics' not supported");
    CHECK(callbacks.empty(), "Arg 'callbacks' not supported");
    CHECK(!options.max_in_memory_batches, "Arg 'max_in_memory' not supported");
    CHECK(!comm, "Arg 'comm' not supported");

    for (uint32_t e = 0; e < epochs; e++) {
      auto dataset = loadData(data, options.batchSize(), /*shuffle*/ true,
                              options.shuffle_config, strong_column_names,
                              weak_column_names, variable_length);

      train(dataset, learning_rate);
    }

    return py::none();
  }

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final {
    CHECK(!sparse_inference, "Sparse inference is not yet supported.");
    CHECK(!return_predicted_class, "return_predicted_class is not supported.");

    auto input = featurizeInput(data::ColumnMap::fromMapInput(sample));
    auto output = _model.forward(input);

    float* data = dense(output->tensor())->data<float>();

    return py::cast(decode(data, top_k.value_or(_default_topk)));
  }

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final {
    CHECK(!sparse_inference, "Sparse inference is not yet supported.");
    CHECK(!return_predicted_class, "return_predicted_class is not supported.");

    auto input = featurizeInput(data::ColumnMap::fromMapInputBatch(sample));
    auto output = _model.forward(input);

    auto tensor = dense(output->tensor());
    float* data = tensor->data<float>();
    size_t batch_size = tensor->shape(0);
    size_t dim = tensor->shape(1);

    uint32_t topk = top_k.value_or(_default_topk);

    std::vector<std::vector<std::pair<uint32_t, double>>> predictions(
        batch_size);

    for (size_t i = 0; i < batch_size; i++) {
      predictions[i] = decode(data + i * dim, topk);
    }

    return py::cast(predictions);
  }

  void introduceDocuments(const dataset::DataSourcePtr& data,
                          const std::vector<std::string>& strong_column_names,
                          const std::vector<std::string>& weak_column_names,
                          std::optional<uint32_t> num_buckets_to_sample,
                          uint32_t num_random_hashes, bool fast_approximation,
                          bool verbose, bool sort_random_hashes) final;

 private:
  data::TransformationPtr coldStartTransform(
      const std::vector<std::string>& strong_cols,
      const std::vector<std::string>& weak_cols,
      std::optional<data::VariableLengthConfig> variable_length,
      bool fast_approximation) const {
    if (fast_approximation) {
      std::vector<std::string> all_columns = weak_cols;
      all_columns.insert(all_columns.end(), strong_cols.begin(),
                         strong_cols.end());
      return std::make_shared<data::StringConcat>(
          all_columns, _text_transform->inputColumn());
    }

    if (variable_length) {
      return std::make_shared<data::VariableLengthColdStart>(
          /* strong_column_names= */ strong_cols,
          /* weak_column_names= */ weak_cols,
          /* output_column_name= */ _text_transform->inputColumn(),
          /* config= */ *variable_length);
    }

    return std::make_shared<data::ColdStartTextAugmentation>(
        /* strong_column_names= */ strong_cols,
        /* weak_column_names= */ weak_cols,
        /* output_column_name= */ _text_transform->inputColumn());
  }

  data::PipelinePtr buildPipeline(
      const std::vector<std::string>& strong_cols,
      const std::vector<std::string>& weak_cols,
      std::optional<data::VariableLengthConfig> variable_length) const {
    auto pipeline = data::Pipeline::make();

    if (!strong_cols.empty() || !weak_cols.empty()) {
      pipeline = pipeline->then(
          coldStartTransform(strong_cols, weak_cols, variable_length,
                             /*fast_approximation=*/false));
    }

    pipeline = pipeline->then(_text_transform)
                   ->then(_entity_parse_transform)
                   ->then(_mach_label_transform);

    return pipeline;
  }

  using Dataset = std::vector<std::pair<smx::VariablePtr, smx::VariablePtr>>;

  Dataset loadData(const dataset::DataSourcePtr& data_source, size_t batch_size,
                   bool shuffle = true,
                   dataset::DatasetShuffleConfig shuffle_config =
                       dataset::DatasetShuffleConfig(),
                   const std::vector<std::string>& strong_cols = {},
                   const std::vector<std::string>& weak_cols = {},
                   std::optional<data::VariableLengthConfig> variable_length =
                       std::nullopt) const {
    auto data_iter = data::CsvIterator::make(data_source, _delimiter);

    auto pipeline = buildPipeline(strong_cols, weak_cols, variable_length);

    data::Loader loader(
        data_iter, pipeline, std::make_shared<data::State>(_mach_index),
        _input_columns, _label_columns, batch_size, shuffle,
        /*verbose=*/true, shuffle_config.min_buffer_size, shuffle_config.seed);

    auto [inputs, labels] = loader.allSmx();

    Dataset dataset;
    for (size_t i = 0; i < inputs.size(); i++) {
      dataset.emplace_back(
          smx::Variable::make(inputs.at(i).at(0), /*requires_grad=*/false),
          smx::Variable::make(labels.at(i).at(0), /*requires_grad=*/false));
    }

    return dataset;
  }

  smx::VariablePtr featurizeInput(data::ColumnMap&& columns) {
    columns = _text_transform->applyStateless(std::move(columns));
    auto tensors = data::toSmxTensors(columns, _input_columns);
    return smx::Variable::make(tensors.at(0), false);
  }

  void train(const Dataset& dataset, float learning_rate) {
    _optimizer.updateLr(learning_rate);

    for (const auto& [x, y] : dataset) {
      _optimizer.zeroGrad();

      auto out = _model.forward(x, y);
      auto loss = smx::binaryCrossEntropy(out, y);
      loss->backward();

      _optimizer.step();
    }
  }

  std::vector<std::pair<uint32_t, double>> decode(float* scores,
                                                  uint32_t top_k) {
    BoltVector vec(/*an=*/nullptr, /*a=*/scores, /*g=*/nullptr,
                   _mach_index->numBuckets());
    return _mach_index->decode(vec, top_k, _default_num_buckets_to_eval);
  }

  data::OutputColumnsList _input_columns = {
      data::OutputColumns(FEATURIZED_INDICES, FEATURIZED_VALUES)};
  data::OutputColumnsList _label_columns = {data::OutputColumns(MACH_LABELS)};

  std::shared_ptr<data::TextTokenizer> _text_transform;
  std::shared_ptr<data::StringToTokenArray> _entity_parse_transform;
  std::shared_ptr<data::MachLabel> _mach_label_transform;
  data::MachIndexPtr _mach_index;
  char _delimiter;

  MachModel _model;
  smx::Adam _optimizer;

  uint32_t _default_topk;
  uint32_t _default_num_buckets_to_eval;
};

}  // namespace thirdai::automl::udt