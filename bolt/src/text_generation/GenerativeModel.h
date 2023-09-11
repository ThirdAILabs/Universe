#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_set.hpp>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <optional>
#include <unordered_set>

namespace thirdai::bolt {

class GenerativeBackend {
 public:
  virtual bolt::TensorPtr nextTokenProbs(
      std::vector<std::vector<uint32_t>> tokens) = 0;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

class GenerativeModel {
 public:
  GenerativeModel(std::shared_ptr<GenerativeBackend> model,
                  std::unordered_set<uint32_t> allowed_repeats,
                  std::unordered_set<uint32_t> punctutation_tokens);

  std::vector<uint32_t> generate(
      const std::vector<uint32_t>& input_tokens, size_t n_predictions,
      size_t beam_width, std::optional<float> temperature = std::nullopt) const;

  void save(const std::string& filename) const;

  static std::shared_ptr<GenerativeModel> load(const std::string& filename);

 private:
  void adjustTokenProbs(const std::vector<uint32_t>& sequence,
                        BoltVector& probs, size_t n_predictions,
                        std::optional<float> temperature) const;

  std::shared_ptr<GenerativeBackend> _model;

  std::unordered_set<uint32_t> _allowed_repeats;
  std::unordered_set<uint32_t> _punctuation_tokens;

  GenerativeModel() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model, _allowed_repeats, _punctuation_tokens);
  }
};

}  // namespace thirdai::bolt