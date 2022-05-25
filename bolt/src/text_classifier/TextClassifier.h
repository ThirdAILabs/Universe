#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <fstream>
#include <memory>

namespace thirdai::bolt {

class TextClassifier {
 public:
  TextClassifier(const std::string& model_size, uint32_t n_classes,
                 uint32_t input_dim = 100000);

  void train(const std::string& filename, uint32_t epochs = 1) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);
    _batch_processor = std::make_shared<dataset::TextClassificationProcessor>();

    dataset::StreamingDataset dataset(data_loader, _batch_processor);

    CategoricalCrossEntropyLoss loss;

    if (epochs == 1) {
      trainOnStreamingDataset(dataset, loss, 0.001);
    } else {
      auto in_memory_dataset = dataset.loadInMemory();

      _model->train(in_memory_dataset.data, in_memory_dataset.labels, loss,
                    0.001, epochs);
    }
  }

  void predict(const std::string& filename,
               const std::string& output_filename) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);

    dataset::StreamingDataset dataset(data_loader, _batch_processor);

    std::ofstream output_file(output_filename);

    MetricAggregator metrics({"categorical_accuracy"});

    _model->initializeNetworkState(dataset.getMaxBatchSize(),
                                   /* force_dense= */ false);
    BoltBatch outputs =
        _model->getOutputs(dataset.getMaxBatchSize(), /* force_dense= */ false);

    while (auto batch = dataset.nextBatch()) {
      _model->processTestBatch(batch->first, outputs, &batch->second,
                               /* output_active_neurons= */ nullptr,
                               /* output_activations = */ nullptr, metrics,
                               /* compute_metrics= */ true);

      for (uint32_t batch_id = 0; batch_id < batch->first.getBatchSize();
           batch_id++) {
        float max_act = 0.0;
        uint32_t pred = 0;
        for (uint32_t i = 0; i < outputs[batch_id].len; i++) {
          if (outputs[batch_id].activations[i] > max_act) {
            max_act = outputs[batch_id].activations[i];
            pred = outputs[batch_id].active_neurons[i];
          }
        }
        output_file << _batch_processor->getClassName(pred);
      }
    }

    output_file.close();
  }

 private:
  void trainOnStreamingDataset(dataset::StreamingDataset& dataset,
                               const LossFunction& loss, float learning_rate) {
    _model->initializeNetworkState(dataset.getMaxBatchSize(), false);

    BoltBatch outputs = _model->getOutputs(dataset.getMaxBatchSize(), false);

    MetricAggregator metrics({});

    uint32_t rehash_batch = 0, rebuild_batch = 0;
    while (auto batch = dataset.nextBatch()) {
      _model->processTrainingBatch(batch->first, outputs, batch->second, loss,
                                   learning_rate, rehash_batch, rebuild_batch,
                                   metrics);
    }
  }

  std::unique_ptr<FullyConnectedNetwork> _model;
  std::shared_ptr<dataset::TextClassificationProcessor> _batch_processor;
};

}  // namespace thirdai::bolt