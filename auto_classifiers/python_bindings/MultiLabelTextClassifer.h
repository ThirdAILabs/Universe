#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_classifiers/python_bindings/AutoClassifierBase.h>
#include <dataset/src/StreamingGenericDatasetLoader.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/pybind11.h>
#include <serialization/Utils.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace thirdai::bolt::python {

/**
 * The MultiLabelTextClassifier takes in data in the form:
 *        <class_id_1>,<class_id_2>,...,<class_id_n>\t<text>.
 * It uses paigrams to featurize the text, and uses sigmoid/bce to handle the
 * variable number of labels. Thresholding is applied to ensure that each
 * prediction has at least one neuron with an activation > the given threshold.
 * Predict and evaluate return numpy arrays of the output activations.
 */
class MultiLabelTextClassifier final
    : public AutoClassifierBase<std::vector<uint32_t>> {
 public:
  explicit MultiLabelTextClassifier(uint32_t n_classes, float threshold = 0.95)
      : AutoClassifierBase(createModel(n_classes), ReturnMode::NumpyArray),
        _threshold(threshold) {}

  void save(const std::string& filename) {
    serialization::saveToFile(*this, filename);
  }

  static std::unique_ptr<MultiLabelTextClassifier> load(
      const std::string& filename) {
    return serialization::loadFromFile(new MultiLabelTextClassifier(),
                                       filename);
  }

  void updateThreshold(float new_threshold) { _threshold = new_threshold; }

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

  void processPredictionBeforeReturning(uint32_t* active_neurons,
                                        float* activations,
                                        uint32_t len) final {
    (void)active_neurons;

    uint32_t max_id = getMaxIndex(activations, len);
    if (activations[max_id] < _threshold) {
      activations[max_id] = _threshold + 0.0001;
    }
  }

  BoltVector featurizeInputForInference(
      const std::vector<uint32_t>& input) final {
    std::string sentence = joinTokensIntoString(input, /* delimiter= */ ' ');

    return dataset::TextEncodingUtils::computePairgrams(
        sentence, dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM);
  }

  std::string getClassName(uint32_t neuron_id) final {
    return std::to_string(neuron_id);
  }

  uint32_t defaultBatchSize() const final { return 2048; }

  bool freezeHashTablesAfterFirstEpoch() const final { return false; }

  bool useSparseInference() const final { return false; }

  std::optional<uint32_t> defaultRebuildHashTablesInterval() const final {
    return 10000;
  }

  std::optional<uint32_t> defaultReconstructHashFunctionsInterval()
      const final {
    return 50000;
  }

  std::vector<std::string> getEvaluationMetrics() const final {
    std::string f_measure = "f_measure(" + std::to_string(_threshold) + ")";
    return {"categorical_accuracy", f_measure};
  }

 private:
  static BoltGraphPtr createModel(uint32_t n_classes) {
    auto model = CommonNetworks::FullyConnected(
        /* input_dim= */ dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM,
        /* layers= */ {FullyConnectedNode::makeDense(
                           /* dim= */ 1024, "relu"),
                       FullyConnectedNode::makeExplicitSamplingConfig(
                           /* dim= */ n_classes,
                           /* sparsity= */ getOutputSparsity(n_classes),
                           /* activation= */ "sigmoid",
                           /* num_tables= */ 64, /* hashes_per_table= */ 4,
                           /* reservoir_size= */ 64)});
    model->compile(std::make_shared<BinaryCrossEntropyLoss>(),
                   /* print_when_done= */ false);

    return model;
  }

  std::unique_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>> getDataset(
      std::shared_ptr<dataset::DataLoader> data_loader) {
    std::vector<dataset::BlockPtr> input_blocks = {
        dataset::PairGramTextBlock::make(/* col= */ 1)};
    std::vector<dataset::BlockPtr> label_blocks = {
        dataset::NumericalCategoricalBlock::make(
            /* col= */ 0,
            /* n_classes= */ _model->outputDim(), /* delimiter= */ ',')};

    return std::make_unique<dataset::StreamingGenericDatasetLoader>(
        /* data_loader= */ std::move(data_loader),
        /* input_blocks= */ input_blocks, /* label_blocks= */ label_blocks,
        /* shuffle= */ true,
        /* shuffle_config= */ dataset::DatasetShuffleConfig(),
        /* has_header= */ false, /* delimiter= */ '\t');
  }

  static float getOutputSparsity(uint32_t output_dim) {
    /*
      For smaller output layers, we return a sparsity
      that puts the sparse dimension between 80 and 160.
    */
    if (output_dim < 450) {
      return 1.0;
    }
    if (output_dim < 900) {
      return 0.2;
    }
    if (output_dim < 1800) {
      return 0.1;
    }
    /*
      For larger layers, we return a sparsity that
      puts the sparse dimension between 100 and 260.
    */
    if (output_dim < 4000) {
      return 0.05;
    }
    if (output_dim < 10000) {
      return 0.02;
    }
    if (output_dim < 20000) {
      return 0.01;
    }
    return 0.05;
  }

  float _threshold;

  // Private constructor for cereal.
  MultiLabelTextClassifier()
      : AutoClassifierBase(nullptr, ReturnMode::NumpyArray) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<AutoClassifierBase>(this), _threshold);
  }
};

}  // namespace thirdai::bolt::python

CEREAL_REGISTER_TYPE(thirdai::bolt::python::MultiLabelTextClassifier)
