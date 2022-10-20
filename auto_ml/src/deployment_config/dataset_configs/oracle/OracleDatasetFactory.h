#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include "Featurizer.h"
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>

namespace thirdai::automl::deployment {

class OracleDatasetFactory final : public DatasetLoaderFactory {
 public:
  explicit OracleDatasetFactory(OracleConfigPtr config, bool parallel,
                                uint32_t text_pairgram_word_limit)
      : _featurizer(std::make_shared<Featurizer>(std::move(config), parallel,
                                                 text_pairgram_word_limit)),
        _context(std::make_shared<TemporalContext>(_featurizer)) {}

  DatasetLoaderPtr getLabeledDatasetLoader(
      std::shared_ptr<dataset::DataLoader> data_loader, bool training) final {
    _featurizer->initializeProcessors(data_loader, *_context);
    return std::make_unique<GenericDatasetLoader>(
        data_loader, _featurizer->getLabeledContextUpdatingProcessor(),
        /* shuffle= */ training);
  }

  std::vector<BoltVector> featurizeInput(const std::string& input) final {
    return _featurizer->featurizeInput(input,
                                       /* should_update_history= */ false);
  }

  std::vector<BoltVector> featurizeInput(const MapInput& input) final {
    return _featurizer->featurizeInput(input,
                                       /* should_update_history= */ false);
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const std::vector<std::string>& inputs) final {
    return _featurizer->featurizeInputBatch(inputs,
                                            /* should_update_history= */ false);
  }

  std::vector<BoltBatch> featurizeInputBatch(
      const MapInputBatch& inputs) final {
    return _featurizer->featurizeInputBatch(inputs,
                                            /* should_update_history= */ false);
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const std::string& sample) final {
    auto input_row = _featurizer->toInputRow(sample);
    return explainImpl(gradients_indices, gradients_ratio, input_row);
  }

  std::vector<dataset::Explanation> explain(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio, const MapInput& sample) final {
    auto input_row = _featurizer->toInputRow(sample);
    return explainImpl(gradients_indices, gradients_ratio, input_row);
  }

  std::vector<bolt::InputPtr> getInputNodes() final {
    return {bolt::Input::make(_featurizer->getInputDim())};
  }

  uint32_t getLabelDim() final { return _featurizer->getLabelDim(); }

  std::vector<std::string> listArtifactNames() const final {
    return {"temporal_context"};
  }

 protected:
  std::optional<Artifact> getArtifactImpl(const std::string& name) const final {
    if (name == "temporal_context") {
      return {_context};
    }
    return nullptr;
  }

 private:
  std::vector<dataset::Explanation> explainImpl(
      const std::optional<std::vector<uint32_t>>& gradients_indices,
      const std::vector<float>& gradients_ratio,
      const std::vector<std::string_view>& input_row) {
    auto result = bolt::getSignificanceSortedExplanations(
        gradients_indices, gradients_ratio, input_row,
        _featurizer->getUnlabeledNonUpdatingProcessor());

    for (auto& response : result) {
      response.column_name =
          _featurizer->colNumToColName(response.column_number);
    }

    return result;
  }

  FeaturizerPtr _featurizer;
  TemporalContextPtr _context;

  // Private constructor for cereal.
  OracleDatasetFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactory>(this), _featurizer,
            _context);
  }
};

class OracleDatasetFactoryConfig final : public DatasetLoaderFactoryConfig {
 public:
  explicit OracleDatasetFactoryConfig(
      HyperParameterPtr<OracleConfigPtr> config,
      HyperParameterPtr<bool> parallel,
      HyperParameterPtr<uint32_t> text_pairgram_word_limit)
      : _config(std::move(config)),
        _parallel(std::move(parallel)),
        _text_pairgram_word_limit(std::move(text_pairgram_word_limit)) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
    auto config = _config->resolve(user_specified_parameters);
    auto parallel = _parallel->resolve(user_specified_parameters);
    auto text_pairgram_word_limit =
        _text_pairgram_word_limit->resolve(user_specified_parameters);

    return std::make_unique<OracleDatasetFactory>(config, parallel,
                                                  text_pairgram_word_limit);
  }

 private:
  HyperParameterPtr<OracleConfigPtr> _config;
  HyperParameterPtr<bool> _parallel;
  HyperParameterPtr<uint32_t> _text_pairgram_word_limit;

  // Private constructor for cereal.
  OracleDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _config,
            _parallel, _text_pairgram_word_limit);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactoryConfig)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OracleDatasetFactory)