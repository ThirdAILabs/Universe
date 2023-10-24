#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <dataset/src/utils/TokenEncoding.h>
#include <proto/encoders.pb.h>
#include <string>

namespace thirdai::dataset {

class TextEncoder {
 public:
  virtual std::vector<uint32_t> encode(const std::vector<uint32_t>& tokens) = 0;

  virtual uint32_t undoEncoding(const std::vector<uint32_t>& tokens,
                                uint32_t index_within_block,
                                uint32_t index_range) {
    (void)tokens;
    (void)index_within_block;
    (void)index_range;
    throw std::invalid_argument(
        "Explanations are not supported for this type of encoding. ");
  }

  virtual proto::data::TextEncoder* toProto() const = 0;

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

  std::vector<uint32_t> encode(const std::vector<uint32_t>& tokens) final {
    return token_encoding::ngrams(tokens, _n);
  }

  uint32_t undoEncoding(const std::vector<uint32_t>& tokens,
                        uint32_t index_within_block,
                        uint32_t index_range) final {
    if (_n != 1) {
      throw std::invalid_argument(
          "Explanations are not supported for this type of encoding.");
    }

    for (const uint32_t token : tokens) {
      // return the first token that mapped to the relevant index
      // TODO() if we ever use rca we should revisit this
      if (token % index_range == index_within_block) {
        return token;
      }
    }

    // should never get here since RCA should have only returned a valid index
    throw std::invalid_argument("Error in RCA");
  }

  proto::data::TextEncoder* toProto() const final {
    proto::data::TextEncoder* encoder = new proto::data::TextEncoder();
    encoder->mutable_ngram()->set_n(_n);
    return encoder;
  }

  uint32_t n() const { return _n; }

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

  std::vector<uint32_t> encode(const std::vector<uint32_t>& tokens) final {
    return token_encoding::pairgrams(tokens);
  }

  proto::data::TextEncoder* toProto() const final {
    proto::data::TextEncoder* encoder = new proto::data::TextEncoder();
    encoder->mutable_pairgram();
    return encoder;
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