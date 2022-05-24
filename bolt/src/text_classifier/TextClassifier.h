#pragma once

#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <memory>

namespace thirdai::bolt {

class TextClassifier {
 public:
  TextClassifier(const std::string& model_size, uint32_t output_dim,
                 uint32_t input_dim = 100000);

  void train(const std::string& filename, uint32_t epochs = 1) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);
    std::shared_ptr<dataset::BatchProcessor> batch_processor =
        std::make_shared<dataset::TextClassificationProcessor>();

    dataset::StreamingDataset dataset(data_loader, batch_processor);

    CategoricalCrossEntropyLoss loss;

    if (epochs == 1) {
      _model->trainOnStreamingDataset(dataset, loss, 0.001);
    } else {
      auto in_memory_dataset = dataset.loadInMemory();

      _model->train(in_memory_dataset.data, in_memory_dataset.labels, loss,
                    0.001, epochs);
    }
  }

 private:
  std::unique_ptr<FullyConnectedNetwork> _model;
};

}  // namespace thirdai::bolt