#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <auto_classifiers/python_bindings/AutoClassifierBase.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::python {

class BinaryTextClassifier final
    : public AutoClassifierBase<std::vector<uint32_t>> {
 public:
  explicit BinaryTextClassifier(uint32_t n_outputs, uint32_t internal_model_dim,
                                std::optional<float> sparsity = std::nullopt,
                                bool use_sparse_inference = true)
      : AutoClassifierBase(
            createAutotunedModel(
                /* internal_model_dim= */ internal_model_dim, n_outputs,
                sparsity,
                /* output_activation= */ ActivationFunction::Sigmoid),
            ReturnMode::NumpyArray),
        _use_sparse_inference(use_sparse_inference) {}

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<BinaryTextClassifier> load(
      const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<BinaryTextClassifier> deserialize_into(
        new BinaryTextClassifier());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

  std::vector<dataset::Explanation> explain(
      const std::vector<uint32_t>& sample,
      std::optional<std::string> target_label) override {
    (void)sample;
    (void)target_label;
    throw std::invalid_argument(
        "Explain method is not yet implemented in BinaryTextClassifier.");
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

  BoltVector featurizeInputForInference(
      const std::vector<uint32_t>& tokens) final {
    std::string sentence = joinTokensIntoString(tokens, /* delimiter= */ ' ');

    return dataset::TextEncodingUtils::computeUnigrams(
        sentence, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    (void)neuron_id;
    throw std::runtime_error(
        "getClassName() is not support for BinaryTextClassifier.");
  }

  uint32_t defaultBatchSize() const final { return 256; }

  bool freezeHashTablesAfterFirstEpoch() const final {
    return _use_sparse_inference;
  }

  bool useSparseInference() const final { return _use_sparse_inference; }

  std::vector<std::string> getEvaluationMetrics() const final { return {}; }

 private:
  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>> getDataset(
      std::shared_ptr<dataset::DataLoader> data_loader) {
    // Because we have n_outputs binary label columns, the text column is starts
    // at _model->outputDim() which is equivalent to n_classes.
    auto batch_processor = dataset::GenericBatchProcessor::make(
        /* input_blocks= */ {dataset::UniGramTextBlock::make(
            /* col= */ _model->outputDim())},
        /* label_blocks= */ {
            dataset::DenseArrayBlock::make(/* start_col= */ 0,
                                           /* dim= */ _model->outputDim())});

    return std::make_unique<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
        std::move(data_loader), batch_processor);
  }

  // Private constructor for cereal.
  BinaryTextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this),
            _use_sparse_inference);
  }

  bool _use_sparse_inference;
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::BinaryTextClassifier)
