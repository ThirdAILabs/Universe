#include "ContextualModel.h"
#include <bolt/src/train/trainer/Dataset.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>

namespace thirdai::bolt {

ContextualModel::ContextualModel(
    bolt::ModelPtr model, dataset::TextGenerationFeaturizerPtr featurizer)
    : _model(std::move(model)), _featurizer(std::move(featurizer)) {}

bolt::TensorPtr ContextualModel::nextTokenProbs(
    std::vector<std::vector<uint32_t>> tokens) {
  auto tensors = _featurizer->featurizeInputBatch(tokens, _model->inputDims());
  return _model->forward(tensors).at(0);
}

metrics::History ContextualModel::train(
    const dataset::DataSourcePtr& train_data, float learning_rate,
    uint32_t epochs, size_t batch_size,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const DistributedCommPtr& comm) {
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
  dataset = {dataset.begin() + 1, dataset.end()};  // Remove 'prompt';

  return {convertDatasets(dataset, _model->inputDims()),
          convertDataset(labels, _model->labelDims().at(0))};
}

}  // namespace thirdai::bolt