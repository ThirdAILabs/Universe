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

  void train(const std::string& filename, uint32_t epochs = 1,
             float learning_rate = 0.001);

  void predict(const std::string& filename, const std::string& output_filename);

 private:
  void trainOnStreamingDataset(dataset::StreamingDataset& dataset,
                               const LossFunction& loss, float learning_rate);

  std::unique_ptr<FullyConnectedNetwork> _model;
  std::shared_ptr<dataset::TextClassificationProcessor> _batch_processor;
};

}  // namespace thirdai::bolt