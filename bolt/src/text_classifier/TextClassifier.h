#pragma once

#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
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
    _batch_processor =
        std::make_shared<dataset::TextClassificationProcessor>(false);

    dataset::StreamingDataset dataset(data_loader, _batch_processor);

    CategoricalCrossEntropyLoss loss;

    if (epochs == 1) {
      _model->trainOnStreamingDataset(dataset, loss, 0.001);
    } else {
      auto in_memory_dataset = dataset.loadInMemory();

      _model->train(in_memory_dataset.data, in_memory_dataset.labels, loss,
                    0.001, epochs);
    }
  }

  void test(const std::string& filename, const std::string& output_filename) {
    std::shared_ptr<dataset::DataLoader> data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);
    _batch_processor->setAsTestData();

    dataset::StreamingDataset dataset(data_loader, _batch_processor);

    auto predictions = _model->testOnStreamingDataset(dataset);

    std::ofstream output_file(output_filename);

    for (const uint32_t pred : predictions) {
      output_file << _batch_processor->getClassName(pred);
    }
    output_file.close();
  }

 private:
  std::unique_ptr<FullyConnectedNetwork> _model;
  std::shared_ptr<dataset::TextClassificationProcessor> _batch_processor;
};

}  // namespace thirdai::bolt