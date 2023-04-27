#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <dataset/src/utils/TokenEncoding.h>
#include <string>

namespace thirdai::dataset {

class TextEncoder {
 public:
  virtual std::vector<uint32_t> apply(
      const std::vector<std::string_view>& tokens) = 0;

  virtual std::string getResponsibleWord(
      const std::vector<std::string_view>& tokens, uint32_t index_within_block,
      uint32_t index_range) {
    (void)tokens;
    (void)index_within_block;
    (void)index_range;
    throw std::invalid_argument(
        "Explanations are not supported for this type of encoding. ");
  }

  virtual ~TextEncoder() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using TextEncoderPtr = std::shared_ptr<TextEncoder>;

class NGramEncoder : public TextEncoder {
 public:
  explicit NGramEncoder(uint32_t n) : _n(n) {}

  static auto make(uint32_t n) { return std::make_shared<NGramEncoder>(n); }

  std::vector<uint32_t> apply(
      const std::vector<std::string_view>& tokens) final {
    return token_encoding::ngrams(token_encoding::hashTokens(tokens), _n);
  }

  std::string getResponsibleWord(const std::vector<std::string_view>& tokens,
                                 uint32_t index_within_block,
                                 uint32_t index_range) final {
    return token_encoding::buildUnigramHashToWordMap(tokens, index_range)
        .at(index_within_block);
  }

 private:
  uint32_t _n;

  NGramEncoder() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this), _n);
  }
};

class PairGramEncoder : public TextEncoder {
 public:
  PairGramEncoder() {}

  static auto make() { return std::make_shared<PairGramEncoder>(); }

  std::vector<uint32_t> apply(
      const std::vector<std::string_view>& tokens) final {
    return token_encoding::pairgrams(token_encoding::hashTokens(tokens));
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this));
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::PairGramEncoder)
CEREAL_REGISTER_TYPE(thirdai::dataset::NGramEncoder)