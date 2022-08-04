#pragma once

#include <cereal/archives/binary.hpp>
#include "AutoClassifierUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/Metric.h>
#include <dataset/src/batch_processors/TextClassificationProcessor.h>
#include <dataset/src/utils/SafeFileIO.h>

namespace thirdai::bolt {

class TextClassifier {
 public:
  TextClassifier(const std::string& model_size, uint32_t n_classes)
      :  // TODO make this value a default for pairgrams/autoclassifiers
        _input_dim(100000),
        _n_classes(n_classes) {
    _model = AutoClassifierUtils::createNetwork(
        /* input_dim= */ _input_dim,
        /* n_classes= */ n_classes, model_size);
    _batch_processor =
        std::make_shared<dataset::TextClassificationProcessor>(_input_dim);
  }

  void train(const std::string& filename, uint32_t epochs,
             float learning_rate) {
    AutoClassifierUtils::train(
        _model, filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            _batch_processor),
        /* epochs= */ epochs,
        /* learning_rate= */ learning_rate);
  }

  void predict(const std::string& filename,
               const std::optional<std::string>& output_filename) {
    AutoClassifierUtils::predict(
        _model, filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            _batch_processor),
        output_filename, _batch_processor->getClassIdToNames());
  }

  std::string predictSingle(const std::string& sentence) {
    BoltVector input = dataset::TextEncodingUtils::computePairgrams(
        /* sentence = */ sentence, /* output_range = */ _input_dim);

    BoltVector output =
        _model->predictSingle({input}, {},
                              /* use_sparse_inference = */ true);

    return _batch_processor->getClassName(
        /* class_id = */ output.getIdWithHighestActivation());
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
    archive(_input_dim, _n_classes, _model, _batch_processor);
  }

  uint32_t _input_dim;
  uint32_t _n_classes;
  std::shared_ptr<BoltGraph> _model;
  std::shared_ptr<dataset::TextClassificationProcessor> _batch_processor;
};

}  // namespace thirdai::bolt