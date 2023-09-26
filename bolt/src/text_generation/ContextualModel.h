#pragma once

#include <cereal/types/base_class.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>

namespace thirdai::bolt {

class ContextualModel final : public GenerativeBackend {
 public:
  ContextualModel(bolt::ModelPtr model,
                  dataset::TextGenerationFeaturizerPtr featurizer);

bolt::TensorPtr nextTokenProbs(
    const std::vector<uint32_t>& prompt, std::vector<std::vector<uint32_t>>& tokens) final;

  metrics::History train(const dataset::DataSourcePtr& train_data,
                         float learning_rate, uint32_t epochs,
                         size_t batch_size,
                         const std::vector<std::string>& train_metrics,
                         const dataset::DataSourcePtr& val_data,
                         const std::vector<std::string>& val_metrics,
                         const DistributedCommPtr& comm) final;

  bolt::ModelPtr getBoltModel() final { return _model; }

 private:
  LabeledDataset loadDataset(const dataset::DataSourcePtr& data,
                             size_t batch_size, bool shuffle) const;

  bolt::ModelPtr _model;
  dataset::TextGenerationFeaturizerPtr _featurizer;

  ContextualModel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<GenerativeBackend>(this), _model, _featurizer);
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::ContextualModel)