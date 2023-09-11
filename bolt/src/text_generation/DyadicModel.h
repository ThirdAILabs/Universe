#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>

namespace thirdai::bolt {

class DyadicModel final : public GenerativeBackend {
 public:
  explicit DyadicModel(bolt::ModelPtr model);

  bolt::TensorPtr nextTokenProbs(
      std::vector<std::vector<uint32_t>> tokens) final;

 private:
  bolt::ModelPtr _model;
  std::shared_ptr<data::DyadicInterval> _dyadic_transform;
  data::OutputColumnsList _bolt_inputs;
  size_t _vocab_size;

  DyadicModel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<GenerativeBackend>(this), _model,
            _dyadic_transform, _bolt_inputs, _vocab_size);
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::DyadicModel)