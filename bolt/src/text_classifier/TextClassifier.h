#pragma once

#include <cereal/archives/binary.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <memory>

namespace thirdai::bolt {

class TextClassifier {
 public:
  TextClassifier(const std::string& model_size, uint32_t n_classes);

  explicit TextClassifier(SequentialConfigList configs);

  void train(const std::string& filename, uint32_t epochs, float learning_rate);

  void predict(const std::string& filename,
               const std::optional<std::string>& output_filename);

  std::shared_ptr<SequentialLayer> getHiddenLayer() {
    return _model->_layers[0];
  }

  void save(const std::string& filename) {
    std::ofstream filestream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<TextClassifier> load(const std::string& filename) {
    std::ifstream filestream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<TextClassifier> deserialize_into(new TextClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
  void trainOnStreamingDataset(dataset::StreamingDataset<BoltBatch>& dataset,
                               const LossFunction& loss, float learning_rate);

  // Private constructor for cereal
  TextClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _batch_processor);
  }

  std::unique_ptr<FullyConnectedNetwork> _model;
  std::shared_ptr<dataset::TextClassificationProcessor> _batch_processor;
};

}  // namespace thirdai::bolt