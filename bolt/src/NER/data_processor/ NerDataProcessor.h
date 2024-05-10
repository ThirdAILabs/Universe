#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_set.hpp>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <licensing/src/CheckLicense.h>
#include <memory>

namespace thirdai::bolt {
class NerDataProcesser : public std::enable_shared_from_this<NerDataProcesser> {
 public:
  NerDataProcesser() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive();
  }
};
}  // namespace thirdai::bolt