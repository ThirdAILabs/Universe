#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/mach/MachBlock.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTMachClassifier final : public UDTBackend {
 public:
  UDTMachClassifier(const data::ColumnDataTypes& input_data_types,
                    const data::UserProvidedTemporalRelationships&
                        temporal_tracking_relationships,
                    const std::string& target_name,
                    const data::CategoricalDataTypePtr& target,
                    uint32_t n_target_classes, bool integer_target,
                    const data::TabularOptions& tabular_options,
                    const std::optional<std::string>& model_config,
                    const config::ArgumentMap& user_args);

  py::object train(
      const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
      const std::optional<ValidationDataSource>& validation,
      std::optional<size_t> batch_size_opt,
      std::optional<size_t> max_in_memory_batches,
      const std::vector<std::string>& metrics,
      const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
      bool verbose, std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose, bool return_metrics) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class) final;

  bolt::BoltGraphPtr model() const final { return _classifier->model(); }

  void setModel(const bolt::BoltGraphPtr& model) final {
    bolt::BoltGraphPtr& curr_model = _classifier->model();
    if (curr_model->outputDim() != curr_model->outputDim()) {
      throw std::invalid_argument("Output dim mismatch in set_model.");
    }
    curr_model = model;
  }

  py::object coldstart(const dataset::DataSourcePtr& data,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& metrics,
                       const std::optional<ValidationDataSource>& validation,
                       const std::vector<bolt::CallbackPtr>& callbacks,
                       std::optional<size_t> max_in_memory_batches,
                       bool verbose) final;

  py::object embedding(const MapInput& sample) final;

  /**
   * This method is still experimental, we should test to see when these
   * embeddings are useful and which tweaks like summing vs averaging and tanh
   * vs reul make a difference.
   */
  py::object entityEmbedding(
      const std::variant<uint32_t, std::string>& label) final;

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

  void setDecodeParams(uint32_t min_num_eval_results,
                       uint32_t top_k_per_eval_aggregation) final;

  void verifyCanDistribute() const final {
    _dataset_factory->verifyCanDistribute();
  }

  void introduceDocuments(const dataset::DataSourcePtr& data) final {
    auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
        data, _dataset_factory->delimiter());

    auto metadata = getColdStartMetaData();

    std::string text_column_name =
        _dataset_factory->inputDataTypes().begin()->first;

    thirdai::data::ColdStartTextAugmentation augmentation(
        /* strong_column_names= */ {"TITLE"},
        /* weak_column_names= */ {"TEXT"},
        /* label_column_name= */ metadata->getLabelColumn(),
        /* output_column_name= */ text_column_name);

    std::vector<std::pair<MapInputBatch, uint32_t>> samples_doc =
        augmentation.getSamplesPerDoc(dataset);

    for (const auto& [samples, doc] : samples_doc) {
      introduce(samples, doc);
    }
  }

  void introduce(const MapInputBatch& samples,
                 const std::variant<uint32_t, std::string>& new_label) final {
    BoltBatch output = _classifier->model()->predictSingleBatch(
        _dataset_factory->featurizeInputBatch(samples),
        /* sparse_inference = */ false);

    // map from output hash to pair of frequency, score
    // a hash appearing more frequently is more indicative than the score but
    // the score is helpful for tiebreaking
    std::unordered_map<uint32_t, uint32_t> candidate_hashes;

    for (const auto& vector : output) {
      auto top_K = vector.findKLargestActivations(
          _mach_label_block->index()->numHashes());

      while (!top_K.empty()) {
        auto [activation, active_neuron] = top_K.top();
        if (!candidate_hashes.count(active_neuron)) {
          // candidate_hashes[active_neuron] = std::make_pair(1, activation);
          candidate_hashes[active_neuron] = 1;
        } else {
          // candidate_hashes[active_neuron].first += 1;
          // candidate_hashes[active_neuron].second += activation;
          candidate_hashes[active_neuron] += 1;
        }
        top_K.pop();
      }
    }

    std::vector<std::pair<uint32_t, uint32_t>> best_hashes(
        candidate_hashes.begin(), candidate_hashes.end());
    std::sort(best_hashes.begin(), best_hashes.end(),
              [](auto& left, auto& right) {
                // auto [left_frequency, left_score] = left.second;
                // auto [right_frequency, right_score] = right.second;
                // if (left_frequency == right_frequency) {
                //   return left_score > right_score;
                // }
                // return left_frequency > right_frequency;
                return left.second > right.second;
              });

    std::vector<uint32_t> new_hashes(_mach_label_block->index()->numHashes());
    for (uint32_t i = 0; i < _mach_label_block->index()->numHashes(); i++) {
      auto [hash, freq_score_pair] = best_hashes[i];
      new_hashes[i] = hash;
    }

    _mach_label_block->index()->manualAdd(variantToString(new_label),
                                          new_hashes);
  }

  void forget(const std::variant<uint32_t, std::string>& label) final {
    _mach_label_block->index()->erase(variantToString(label));

    if (_mach_label_block->index()->numElements() == 0) {
      std::cout << "Warning. Every learned class has been forgotten. The model "
                   "will currently return nothing on calls to evaluate, "
                   "predict, or predictBatch."
                << std::endl;
    }
  }

  TextEmbeddingModelPtr getTextEmbeddingModel(
      const std::string& activation_func, float distance_cutoff) const final;

 private:
  bool integerTarget() const {
    return static_cast<bool>(
        dataset::mach::asNumericIndex(_mach_label_block->index()));
  }

  cold_start::ColdStartMetaDataPtr getColdStartMetaData() final {
    return std::make_shared<cold_start::ColdStartMetaData>(
        /* label_delimiter = */ _mach_label_block->delimiter(),
        /* label_column_name = */ _mach_label_block->columnName(),
        /* integer_target = */
        integerTarget());
  }

  std::string variantToString(
      const std::variant<uint32_t, std::string>& variant);

  static uint32_t autotuneMachOutputDim(uint32_t n_target_classes) {
    // TODO(david) update this
    if (n_target_classes < defaults::MACH_MIN_TARGET_CLASSES) {
      return n_target_classes;
    }
    return n_target_classes / defaults::MACH_DEFAULT_OUTPUT_RANGE_SCALEDOWN;
  }

  static uint32_t autotuneMachNumHashes(uint32_t n_target_classes,
                                        uint32_t output_range) {
    // TODO(david) update this
    (void)n_target_classes;
    (void)output_range;
    return defaults::MACH_DEFAULT_NUM_REPETITIONS;
  }

  UDTMachClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  std::shared_ptr<utils::Classifier> _classifier;
  dataset::mach::MachBlockPtr _mach_label_block;
  data::TabularDatasetFactoryPtr _dataset_factory;
  uint32_t _min_num_eval_results;
  uint32_t _top_k_per_eval_aggregation;
};

}  // namespace thirdai::automl::udt