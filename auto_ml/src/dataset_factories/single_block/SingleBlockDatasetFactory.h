#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <utils/StringManipulation.h>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace thirdai::automl::data {

class SingleBlockDatasetFactory final : public DatasetLoaderFactory {
 public:
  SingleBlockDatasetFactory(dataset::BlockPtr data_block,
                            dataset::BlockPtr unlabeled_data_block,
                            dataset::BlockPtr label_block, bool shuffle,
                            char delimiter, bool has_header)
      : _labeled_featurizer(std::make_shared<dataset::TabularFeaturizer>(
            std::vector<dataset::BlockPtr>{std::move(data_block)},
            std::vector<dataset::BlockPtr>{std::move(label_block)},
            /* has_header= */ has_header, delimiter)),
        _unlabeled_featurizer(std::make_shared<dataset::TabularFeaturizer>(
            std::vector<dataset::BlockPtr>{std::move(unlabeled_data_block)},
            std::vector<dataset::BlockPtr>{},
            /* has_header= */ has_header, delimiter)),
        _shuffle(shuffle),
        _delimiter(delimiter) {}

  dataset::DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataSource> data_source, bool training) final {
    return std::make_unique<dataset::DatasetLoader>(
        data_source, _labeled_featurizer, _shuffle && training);
  }

  std::vector<BoltVector> featurizeInput(const std::string& input) final;

  std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs) final;

  uint32_t labelToNeuronId(std::variant<uint32_t, std::string> label) final;

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const std::string& sample) final;

  std::vector<uint32_t> getInputDims() final {
    return {_unlabeled_featurizer->getInputDim()};
  }

  uint32_t getLabelDim() final { return _labeled_featurizer->getLabelDim(); }

  bool hasTemporalTracking() const final { return false; }

 private:
  dataset::TabularFeaturizerPtr _labeled_featurizer;
  dataset::TabularFeaturizerPtr _unlabeled_featurizer;
  bool _shuffle;
  char _delimiter;

  // Private constructor for cereal.
  SingleBlockDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _labeled_featurizer,
            _unlabeled_featurizer, _shuffle, _delimiter);
  }
};

}  // namespace thirdai::automl::data
