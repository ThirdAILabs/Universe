#pragma once

#include <cereal/types/base_class.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>

namespace thirdai::bolt {

class ContextualModel final : public GenerativeBackend {
 public:
  ContextualModel(bolt::ModelPtr model,
                  dataset::TextGenerationFeaturizerPtr featurizer);

  bolt::TensorPtr nextTokenProbs(
      std::vector<std::vector<uint32_t>> tokens) final;

 private:
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