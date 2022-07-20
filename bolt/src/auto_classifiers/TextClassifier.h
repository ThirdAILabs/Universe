#pragma once

#include <cereal/archives/binary.hpp>
#include "AutoClassifierUtils.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/TextClassificationProcessor.h>
#include <dataset/src/utils/SafeFileIO.h>

namespace thirdai::bolt {

class TextClassifier {
 public:
  TextClassifier(const std::string& model_size, uint32_t n_classes) {
    uint32_t input_dim = 100000;
    _model = AutoClassifierUtils::createNetwork(
        /* input_dim */ input_dim,
        /* n_classes */ n_classes, model_size);
    _batch_processor =
        std::make_shared<dataset::TextClassificationProcessor>(input_dim);
  }

  void train(const std::string& filename, uint32_t epochs,
             float learning_rate) {
    AutoClassifierUtils::train(
        _model, filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch>>(
            _batch_processor),
        /* epochs */ epochs,
        /* learning_rate */ learning_rate);
  }

  void predict(const std::string& filename,
               const std::optional<std::string>& output_filename) {
    AutoClassifierUtils::predict(
        _model, filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch>>(
            _batch_processor),
        output_filename, _batch_processor->getClassIdToNames());
  }

  std::string predictSingle(const std::string& sentence) {
    BoltVector pairgrams_vec = dataset::PairgramHasher::computePairgrams(
        /*sentence = */ sentence, /*output_range = */ _model->getInputDim());
    BoltVector output =
        BoltVector(/*l = */ _model->getOutputDim(), /*is_dense = */ true);
    _model->initializeNetworkState(/*batch_size = */ 1,
                                   /*use_sparsity = */ true);
    _model->forward(/*batch_index = */ 0, /*input = */ pairgrams_vec, output,
                    /*labels = */ nullptr);
    return _batch_processor->getClassName(
        /*class_id = */ output.getIdWithHighestActivation());
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<TextClassifier> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<TextClassifier> deserialize_into(new TextClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

 private:
  // Private constructor for cereal
  TextClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _batch_processor);
  }

  std::shared_ptr<FullyConnectedNetwork> _model;
  std::shared_ptr<dataset::TextClassificationProcessor> _batch_processor;
};

}  // namespace thirdai::bolt