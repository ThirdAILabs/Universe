#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <optional>
#include <vector>

namespace thirdai::automl::deployment {

class ValidationOptions {
 public:
  ValidationOptions(std::string filename, std::vector<std::string> metrics,
                    std::optional<uint32_t> interval, bool use_sparse_inference)
      : _filename(std::move(filename)),
        _metrics(std::move(metrics)),
        _interval(interval),
        _use_sparse_inference(use_sparse_inference) {}

  const std::string& filename() const { return _filename; }

  // TODO(Nicholas): Refactor ValidationContext to use an optional to indicate
  // validation batch frequency instead of having 0 be a special value.
  uint32_t interval() const { return _interval.value_or(0); }

  bolt::EvalConfig validationConfig() const {
    bolt::EvalConfig val_config =
        bolt::EvalConfig::makeConfig().withMetrics(_metrics);

    if (_use_sparse_inference) {
      val_config.enableSparseInference();
    }

    return val_config;
  }

 private:
  std::string _filename;
  std::vector<std::string> _metrics;
  std::optional<uint32_t> _interval;
  bool _use_sparse_inference;
};

class TrainOptions {
 public:
  explicit TrainOptions(dataset::DataLoaderPtr data_source)
      : _train_data_source(std::move(data_source)),
        _learning_rate(0.001),
        _epochs(3) {}

  void setLearningRate(float learning_rate) { _learning_rate = learning_rate; }

  void setEpochs(uint32_t epochs) { _epochs = epochs; }

  void setValidation(ValidationOptions validation_options) {
    _validation = std::move(validation_options);
  }

  void setCallbacks(std::vector<bolt::CallbackPtr> callbacks) {
    _callbacks = std::move(callbacks);
  }

  void setMaxInMemoryBatches(uint32_t max_batches) {
    _max_in_memory_batches = max_batches;
  }

  dataset::DataLoaderPtr trainData() const { return _train_data_source; }

  bolt::TrainConfig getTrainConfig(
      DatasetLoaderFactoryPtr& dataset_factory) const {
    auto train_config = bolt::TrainConfig::makeConfig(_learning_rate, _epochs);

    train_config.withCallbacks(_callbacks);

    if (_validation) {
      auto file_loader = dataset::SimpleFileDataLoader::make(
          _validation->filename(), /* target_batch_size= */ 2048);

      auto dataset_loader =
          dataset_factory->getLabeledDatasetLoader(std::move(file_loader),
                                                   /* training= */ false);

      auto [val_data, val_labels] =
          dataset_loader->loadInMemory(std::numeric_limits<uint32_t>::max())
              .value();

      train_config.withValidation(
          /* validation_data= */ val_data,
          /* validation_labels= */ val_labels,
          /* eval_config= */ _validation->validationConfig(),
          /* validation_frequency= */ _validation->interval());
    }

    return train_config;
  }

  std::optional<uint32_t> maxInMemoryBatches() const {
    return _max_in_memory_batches;
  }

  bool streaming() const { return _max_in_memory_batches.has_value(); }

 private:
  // Training dataset.
  dataset::DataLoaderPtr _train_data_source;

  // To be used to construct the train config.
  float _learning_rate;
  uint32_t _epochs;
  std::optional<ValidationOptions> _validation;
  std::vector<bolt::CallbackPtr> _callbacks;

  // Optional training parameter to enable streaming.
  std::optional<uint32_t> _max_in_memory_batches;
};

}  // namespace thirdai::automl::deployment