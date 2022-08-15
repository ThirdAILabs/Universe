#pragma once

#include <bolt/src/auto_classifiers/AutoClassifierBase.h>
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/encodings/categorical/CategoricalMultiLabel.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

class WayfairClassifier {
 public:
  explicit WayfairClassifier(uint32_t n_classes)
  : _n_classes(n_classes) {
    buildBatchProcessors(n_classes);

    assert(n_classes == _processor->getLabelDim());
    
    std::vector<std::pair<uint32_t, float>> hidden_layer_config = {{1000, 1.0}};

    _classifier = std::make_unique<AutoClassifierBase>(
      /* input_dim= */ _processor->getInputDim(),
      /* hidden_layer_configs= */ hidden_layer_config,
      /* output_layer_size= */ n_classes,
      /* output_layer_sparsity= */ n_classes >= 500 ? 0.1 : 1
    );
  }

  void train(const std::string& filename,
             uint32_t epochs,
             float learning_rate,
             float fmeasure_threshold) {
    std::stringstream metric_ss;
    metric_ss << "f_measure(" << fmeasure_threshold << ")";
    std::vector<std::string> metrics = {metric_ss.str()};

    _classifier->train(
        filename,
        std::static_pointer_cast<dataset::BatchProcessor<BoltBatch, BoltBatch>>(
            _processor),
        epochs, learning_rate, metrics);
  }

  void predict(const std::string& filename,
               float fmeasure_threshold,
               const std::optional<std::string>& output_filename) {
    std::stringstream metric_ss;
    metric_ss << "f_measure(" << fmeasure_threshold << ")";
    std::vector<std::string> metrics = {metric_ss.str()};
    
    // All class names are strings of the IDs themselves since the 
    // labels are integers.
    std::vector<std::string> class_id_to_name(_n_classes);
    for (uint32_t id = 0; id < _n_classes; id++) {
      std::stringstream id_ss;
      id_ss << id;
      class_id_to_name[id] = id_ss.str();
    }

    _classifier->predict(
        filename,
        _processor,
        output_filename, class_id_to_name, metrics);
  }

  BoltVector predictSingle(const std::vector<uint32_t>& tokens, float threshold = 0.9) {

    std::string sentence = tokensToSentence(tokens);
    // The following step must be separate from the above 
    // because we need to keep the sentence in scope and alive.
    auto sample = sentenceToSample(sentence); 
    
    BoltVector input_vector;
    auto input = _single_inference_processor->makeInputVector(sample, input_vector);

    BoltVector output =
        _classifier->predictSingle({input_vector}, {},
                                   /* use_sparse_inference = */ false);
    
    assert(output.isDense());
    auto max_id = output.getIdWithHighestActivation();
    if (output.activations[max_id] < threshold) {
      output.activations[max_id] = threshold;
    }

    return output;
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<WayfairClassifier> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<WayfairClassifier> deserialize_into(
        new WayfairClassifier());
    iarchive(*deserialize_into);
    deserialize_into->buildBatchProcessors(deserialize_into->_n_classes);
    return deserialize_into;
  }

 private:
  void buildBatchProcessors(uint32_t n_classes) {
    auto multi_label_encoding = std::make_shared<dataset::CategoricalMultiLabel>(/* n_classes= */ n_classes, /* delimiter= */ ',');
    auto label_block = std::make_shared<dataset::CategoricalBlock>(/* col= */ 0, /* encoding= */ multi_label_encoding);
    std::vector<std::shared_ptr<dataset::Block>> label_blocks = {label_block};

    auto pairgram_encoding = std::make_shared<dataset::PairGram>(/* dim= */ 100000);
    auto input_block = std::make_shared<dataset::TextBlock>(/* col= */ 1, /* encoding= */ pairgram_encoding);
    std::vector<std::shared_ptr<dataset::Block>> input_blocks = {input_block};
    auto single_inference_input_block = std::make_shared<dataset::TextBlock>(/* col= */ 0, /* encoding= */ pairgram_encoding); // No label column during single inference
    std::vector<std::shared_ptr<dataset::Block>> single_inference_input_blocks = {single_inference_input_block};
    
    _processor = std::make_shared<dataset::GenericBatchProcessor>(
      input_blocks, label_blocks,
      /* has_header= */ false, /* delimiter= */ '\t'
    );

    _single_inference_processor = std::make_shared<dataset::GenericBatchProcessor>(
      single_inference_input_blocks, /* label_blocks= */ std::vector<std::shared_ptr<dataset::Block>>(), // no label block for single inference
      /* has_header= */ false, /* delimiter= */ '\t'
    );
  }

  static std::string tokensToSentence(const std::vector<uint32_t>& tokens) {
    std::stringstream sentence_ss;
    std::string delim = "";
    for (auto token : tokens) {
      sentence_ss << delim << token;
      delim = " ";
    }
    return sentence_ss.str();
  }

  static std::vector<std::string_view> sentenceToSample(const std::string& sentence) {
    std::vector<std::string_view> sample;
    sample.push_back(std::string_view(sentence.data(), sentence.size()));
    return sample;
  }

  // Private constructor for cereal
  WayfairClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_n_classes, _classifier);
  }
 
 protected:
  uint32_t _n_classes;
  std::shared_ptr<dataset::GenericBatchProcessor> _processor;
  std::shared_ptr<dataset::GenericBatchProcessor> _single_inference_processor;
  std::unique_ptr<AutoClassifierBase> _classifier;
};

} // namespace thirdai::bolt