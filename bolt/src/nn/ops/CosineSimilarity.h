#pragma once
#include <cereal/access.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

enum class ClippingMode { Identity = 0, Sigmoid = 1, LinearScaling = 2 };

inline ClippingMode getClippingMode(
    const std::optional<std::string>& clipping_mode) {
  std::string clipping_mode_str = clipping_mode.value_or("Identity");
  if (clipping_mode_str == "Identity") {
    return ClippingMode::Identity;
  }
  if (clipping_mode_str == "Sigmoid") {
    return ClippingMode::Sigmoid;
  }
  if (clipping_mode_str == "LinearScaling") {
    return ClippingMode::LinearScaling;
  }
  throw std::invalid_argument(
      "Invalid Clipping Mode specified for CosineSimilarity OP");
}

class CosineSimilarity final
    : public Op,
      public std::enable_shared_from_this<CosineSimilarity> {
 public:
  static auto make(std::optional<const std::string>& clipping_mode) {
    ClippingMode clipping_mode_enum = getClippingMode(clipping_mode);
    return std::make_shared<CosineSimilarity>(clipping_mode_enum);
  }

  explicit CosineSimilarity(ClippingMode clipping_mode)
      : _clipping_mode(clipping_mode) {}

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(ComputationPtr lhs, ComputationPtr rhs);

 private:
  CosineSimilarity() {}

  friend class cereal::access;
  ClippingMode _clipping_mode;

  template <class Archive>
  void serialize(Archive& archive);

  static float magnitude(const BoltVector& a);

  static float denseDenseSim(const BoltVector& a, const BoltVector& b);

  static float denseSparseSim(const BoltVector& a, const BoltVector& b);

  static float sparseSparseSim(const BoltVector& a, const BoltVector& b);

  static void denseDenseBackprop(float grad, float cos_sim, BoltVector& a,
                                 BoltVector& b);

  static void denseSparseBackprop(float grad, float cos_sim, BoltVector& a,
                                  BoltVector& b);

  static void sparseSparseBackprop(float grad, float cos_sim, BoltVector& a,
                                   BoltVector& b);
};

using CosineSimilarityPtr = std::shared_ptr<CosineSimilarity>;

}  // namespace thirdai::bolt
