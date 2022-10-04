#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <auto_classifiers/python_bindings/AutoClassifierBase.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::python {

/**
 * The TextClassifier takes in data in the form:
 *        <class_name>,<text>.
 * It uses paigrams to featurize the text and automatically maps the class names
 * to output neurons. Evaluate and predict return lists of strings of the
 * predicted class names.
 */
class TextClassifier final : public AutoClassifierBase<std::string> {
 public:
  TextClassifier(uint32_t internal_model_dim, uint32_t n_classes)
      : AutoClassifierBase(
            createAutotunedModel(
                internal_model_dim, n_classes,
                /* sparsity= */ std::nullopt,
                /* output_activation= */ ActivationFunction::Softmax),
            ReturnMode::ClassName) {
    _label_id_lookup = dataset::ThreadSafeVocabulary::make(n_classes);
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

  std::vector<dataset::Explanation> explain(
      const std::string& sample,
      std::optional<std::string> target_label) override {
    (void)sample;
    (void)target_label;
    throw std::invalid_argument("Explain method is not yet implemented in TextClassifier.");
  }

 protected:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getTrainingDataset(std::shared_ptr<dataset::DataLoader> data_loader,
                     std::optional<uint64_t> max_in_memory_batches) final {
    (void)max_in_memory_batches;
    return getDataset(data_loader);
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
  getEvalDataset(std::shared_ptr<dataset::DataLoader> data_loader) final {
    return getDataset(data_loader);
  }

  BoltVector featurizeInputForInference(const std::string& input_str) final {
    return dataset::TextEncodingUtils::computePairgrams(
        input_str, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    return _label_id_lookup->getString(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 256; }

  bool freezeHashTablesAfterFirstEpoch() const final { return true; }

  bool useSparseInference() const final { return true; }

  std::vector<std::string> getEvaluationMetrics() const final {
    return {"categorical_accuracy"};
  }

 private:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>> getDataset(
      std::shared_ptr<dataset::DataLoader> data_loader) {
    auto label_block = dataset::StringLookupCategoricalBlock::make(
        /* col= */ 0, _label_id_lookup);
    auto batch_processor = dataset::GenericBatchProcessor::make(
        {dataset::PairGramTextBlock::make(/* col= */ 1)}, {label_block});

    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), batch_processor);
  }

  // Private constructor for cereal.
  TextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray),
        _label_id_lookup(nullptr) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this), _label_id_lookup);
  }

  dataset::ThreadSafeVocabularyPtr _label_id_lookup;
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::TextClassifier)
