#include "UDTMachSmx.h"
#include <_types/_uint32_t.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/SmxTensorConversion.h>
#include <data/src/transformations/Pipeline.h>
#include <smx/src/autograd/Variable.h>

namespace thirdai::automl::udt {

py::object UDTMachSmx::train(const dataset::DataSourcePtr& data,
                             float learning_rate, uint32_t epochs,
                             const std::vector<std::string>& train_metrics,
                             const dataset::DataSourcePtr& val_data,
                             const std::vector<std::string>& val_metrics,
                             const std::vector<CallbackPtr>& callbacks,
                             TrainOptions options,
                             const bolt::DistributedCommPtr& comm) {
  CHECK(train_metrics.empty(), "Arg 'train_metrics' not supported.");
  CHECK(!val_data, "Arg 'val_data' not supported");
  CHECK(val_metrics.empty(), "Arg 'val_metrics' not supported");
  CHECK(callbacks.empty(), "Arg 'callbacks' not supported");
  CHECK(!options.max_in_memory_batches, "Arg 'max_in_memory' not supported");
  CHECK(!comm, "Arg 'comm' not supported");

  auto dataset =
      loadTrainingData(data, options.batchSize(), options.shuffle_config);

  for (uint32_t e = 0; e < epochs; e++) {
    train(dataset, learning_rate);
  }

  return py::none();
}

py::object UDTMachSmx::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  CHECK(train_metrics.empty(), "Arg 'train_metrics' not supported.");
  CHECK(!val_data, "Arg 'val_data' not supported");
  CHECK(val_metrics.empty(), "Arg 'val_metrics' not supported");
  CHECK(callbacks.empty(), "Arg 'callbacks' not supported");
  CHECK(!options.max_in_memory_batches, "Arg 'max_in_memory' not supported");
  CHECK(!comm, "Arg 'comm' not supported");

  auto cold_start =
      coldStartTransform(strong_column_names, weak_column_names,
                         variable_length, /*fast_approximation=*/false);

  for (uint32_t e = 0; e < epochs; e++) {
    auto dataset = loadTrainingData(data, options.batchSize(),
                                    options.shuffle_config, cold_start);

    train(dataset, learning_rate);
  }

  return py::none();
}

py::object UDTMachSmx::predict(const MapInput& sample, bool sparse_inference,
                               bool return_predicted_class,
                               std::optional<uint32_t> top_k) {
  CHECK(!sparse_inference, "Sparse inference is not yet supported.");
  CHECK(!return_predicted_class, "return_predicted_class is not supported.");

  auto input = featurizeInput(data::ColumnMap::fromMapInput(sample));
  auto output = _model.forward(input);

  float* data = dense(output->tensor())->data<float>();

  return py::cast(decode(data, top_k.value_or(_default_topk)));
}

py::object UDTMachSmx::predictBatch(const MapInputBatch& sample,
                                    bool sparse_inference,
                                    bool return_predicted_class,
                                    std::optional<uint32_t> top_k) {
  CHECK(!sparse_inference, "Sparse inference is not yet supported.");
  CHECK(!return_predicted_class, "return_predicted_class is not supported.");

  auto input = featurizeInput(data::ColumnMap::fromMapInputBatch(sample));
  auto output = _model.forward(input);

  auto tensor = dense(output->tensor());
  float* data = tensor->data<float>();
  size_t batch_size = tensor->shape(0);
  size_t dim = tensor->shape(1);

  uint32_t topk = top_k.value_or(_default_topk);

  std::vector<std::vector<std::pair<uint32_t, double>>> predictions(batch_size);

  for (size_t i = 0; i < batch_size; i++) {
    predictions[i] = decode(data + i * dim, topk);
  }

  return py::cast(predictions);
}

data::TransformationPtr UDTMachSmx::coldStartTransform(
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols,
    std::optional<data::VariableLengthConfig> variable_length,
    bool fast_approximation) const {
  if (fast_approximation) {
    std::vector<std::string> all_columns = weak_cols;
    all_columns.insert(all_columns.end(), strong_cols.begin(),
                       strong_cols.end());
    return std::make_shared<data::StringConcat>(all_columns,
                                                _text_transform->inputColumn());
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

UDTMachSmx::TrainingDataset UDTMachSmx::loadTrainingData(
    const dataset::DataSourcePtr& data_source, size_t batch_size,
    dataset::DatasetShuffleConfig shuffle_config,
    const data::TransformationPtr& cold_start) const {
  auto data_iter = data::CsvIterator::make(data_source, _delimiter);

  auto pipeline = data::Pipeline::make();
  if (cold_start) {
    pipeline = pipeline->then(cold_start);
  }
  pipeline = pipeline->then(_text_transform)
                 ->then(_entity_parse_transform)
                 ->then(_mach_label_transform);

  data::Loader loader(
      data_iter, pipeline, std::make_shared<data::State>(_mach_index),
      _input_columns, _label_columns, batch_size, /*shuffle=*/true,
      /*verbose=*/true, shuffle_config.min_buffer_size, shuffle_config.seed);

  auto [inputs, labels] = loader.allSmx();

  TrainingDataset dataset;
  for (size_t i = 0; i < inputs.size(); i++) {
    dataset.emplace_back(
        smx::Variable::make(inputs.at(i).at(0), /*requires_grad=*/false),
        smx::Variable::make(labels.at(i).at(0), /*requires_grad=*/false));
  }

  return dataset;
}

UDTMachSmx::EvalDataset UDTMachSmx::loadEvalData(
    const dataset::DataSourcePtr& data_source, size_t batch_size,
    const data::TransformationPtr& cold_start) const {
  auto columns = data::CsvIterator::all(data_source, _delimiter);

  auto pipeline = data::Pipeline::make();
  if (cold_start) {
    pipeline = pipeline->then(cold_start);
  }
  pipeline = pipeline->then(_text_transform)->then(_entity_parse_transform);

  columns = pipeline->applyStateless(columns);

  auto inputs = data::toSmxTensorBatches(columns, _input_columns, batch_size);

  auto labels = columns.getArrayColumn<uint32_t>(MACH_DOC_IDS);

  size_t row_cnt = 0;
  EvalDataset dataset;
  dataset.reserve(inputs.size());
  for (const auto& input : inputs) {
    size_t batch_size = input[0]->shape(0);

    std::vector<std::vector<uint32_t>> batch_labels;
    batch_labels.reserve(batch_size);
    for (size_t i = row_cnt; i < row_cnt + batch_size; i++) {
      batch_labels.push_back(labels->row(i).toVector());
    }

    dataset.emplace_back(smx::Variable::make(input[0], /*requires_grad=*/false),
                         batch_labels);
  }

  return dataset;
}

void UDTMachSmx::train(const UDTMachSmx::TrainingDataset& dataset,
                       float learning_rate) {
  _optimizer.updateLr(learning_rate);

  if (!dataset.empty()) {
    size_t batch_size = dataset.at(0).first->tensor()->shape(0);
    _model.out->autotuneHashTableRebuild(dataset.size(), batch_size);
  }

  for (const auto& [x, y] : dataset) {
    _optimizer.zeroGrad();

    auto out = _model.forward(x, y);
    auto loss = smx::binaryCrossEntropy(out, y);
    loss->backward();

    _optimizer.step();
  }
}

}  // namespace thirdai::automl::udt