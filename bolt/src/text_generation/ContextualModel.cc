#include "ContextualModel.h"
#include <bolt/src/train/trainer/Dataset.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <stdexcept>

namespace thirdai::bolt {

ContextualModel::ContextualModel(
    bolt::ModelPtr model, dataset::TextGenerationFeaturizerPtr featurizer)
    : _model(std::move(model)), _featurizer(std::move(featurizer)) {}

bolt::TensorPtr ContextualModel::nextTokenProbs(
    std::vector<uint32_t>& prompt, std::vector<std::vector<uint32_t>> tokens) {
  auto tensors =
      _featurizer->featurizeInputBatch(prompt, tokens, _model->inputDims());
  return _model->forward(tensors).at(0);
}

metrics::History ContextualModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    std::optional<size_t> max_in_memory_batches,
    std::optional<size_t> rows_per_load,
    const DistributedCommPtr& comm) {
  (void)rows_per_load;
  if (max_in_memory_batches) {
    throw std::runtime_error(
        "Streaming data loading is not implemented for ContextualModel");
  }

  auto train_dataset = loadDataset(train_data, batch_size, /* shuffle= */ true);
  auto val_dataset = loadDataset(val_data, batch_size, /* shuffle= */ false);

  Trainer trainer(_model);

  return trainer.train_with_metric_names(
      train_dataset, learning_rate, epochs, train_metrics, val_dataset,
      val_metrics, /* steps_per_validation= */ std::nullopt,
      /* use_sparsity_in_validation= */ false, /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ false, /* verbose= */ true,
      /* logging_interval= */ std::nullopt, comm);
}

LabeledDataset ContextualModel::loadDataset(const dataset::DataSourcePtr& data,
                                            size_t batch_size,
                                            bool shuffle) const {
  dataset::DatasetLoader loader(data, _featurizer, shuffle);

  auto dataset = loader.loadAll(batch_size);
  auto labels = dataset.back();
  dataset.pop_back();
  size_t input_size = _model->inputs().size();

  if (input_size == 3) {
    dataset = {dataset.begin() + 1, dataset.end()};  // Remove 'prompt';
  } else if (input_size < 3 || input_size > 4) {
    throw std::invalid_argument(
        "Unsupported model input size (" + std::to_string(input_size) +
        "). Featurization logic doesnot fits for this model's inputs.");
  }

  return {convertDatasets(dataset, _model->inputDims()),
          convertDataset(labels, _model->labelDims().at(0))};
}

}  // namespace thirdai::bolt