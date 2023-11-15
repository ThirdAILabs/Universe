#pragma once

#include <bolt/src/nn/loss/Loss.h>

namespace thirdai::bolt {

class ExternalLoss final : public Loss {
 public:
  ExternalLoss(ComputationPtr output, ComputationPtr external_gradients)
      : _output(std::move(output)),
        _external_gradients(std::move(external_gradients)) {}

  void gradients(uint32_t index_in_batch, uint32_t batch_size) const final;

  float loss(uint32_t index_in_batch) const final;

  ComputationList outputsUsed() const final { return {_output}; }

  ComputationList labels() const final { return {_external_gradients}; }

  ar::ConstArchivePtr toArchive() const final;

  static std::shared_ptr<ExternalLoss> fromArchive(
      const ar::Archive& archive,
      const std::unordered_map<std::string, ComputationPtr>& computations);

  static std::string type() { return "external_loss"; }

 private:
  ComputationPtr _output;
  ComputationPtr _external_gradients;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using ExternalLossPtr = std::shared_ptr<ExternalLoss>;

}  // namespace thirdai::bolt