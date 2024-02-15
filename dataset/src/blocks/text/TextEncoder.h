#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

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

  virtual ar::ConstArchivePtr toArchive() const = 0;

  static std::shared_ptr<TextEncoder> fromArchive(const ar::Archive& archive);

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

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    map->set("n", ar::u64(_n));
    return map;
  }

  uint32_t n() const { return _n; }

  static std::string type() { return "ngram"; }

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

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    return map;
  }

  static std::string type() { return "pairgram"; }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this));
  }
};

class FixedDimEncoder : public TextEncoder {
 public:
  explicit FixedDimEncoder(uint32_t max_tokens) : _max_tokens(max_tokens) {}

  static auto make(uint32_t max_tokens) {
    return std::make_shared<FixedDimEncoder>(max_tokens);
  }

  std::vector<uint32_t> encode(const std::vector<uint32_t>& tokens) final {
    auto encodings = token_encoding::ngrams(tokens, 1);
    if (encodings.size() < _max_tokens) {
      auto pairgram_encodings = token_encoding::pairgrams(tokens);
      std::copy(pairgram_encodings.begin(), pairgram_encodings.end(),
                std::back_inserter(encodings));
    }

    if (encodings.size() < _max_tokens) {
      auto three_gram_encodings = token_encoding::ngrams(tokens, 3);
      std::copy(three_gram_encodings.begin(), three_gram_encodings.end(),
                std::back_inserter(encodings));
    }

    return encodings;
  }

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    map->set("_max_tokens", ar::u64(_max_tokens));
    return map;
  }

  static std::string type() { return "fixed_dim"; }

 private:
  friend class cereal::access;
  FixedDimEncoder() {}
  uint32_t _max_tokens;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this), _max_tokens);
  }
};

class CompositeEncoder : public TextEncoder {
 public:
  CompositeEncoder(uint32_t max_tokens,
                   const std::vector<TextEncoderPtr>& encoders,
                   std::string sampling_strategy)
      : _max_tokens(max_tokens),
        _encoders(encoders),
        _sampling_strategy(std::move(sampling_strategy)) {
    if (_sampling_strategy != "fifo") {
      throw std::invalid_argument(
          "Invalid Sampling Strategy for CompositeEncoder: " +
          _sampling_strategy);
    }
  }

  static auto make(uint32_t max_tokens,
                   const std::vector<TextEncoderPtr>& encoders,
                   std::string sampling_strategy) {
    return std::make_shared<CompositeEncoder>(max_tokens, encoders,
                                              sampling_strategy);
  }

  std::vector<uint32_t> encode(const std::vector<uint32_t>& tokens) final {
    std::vector<uint32_t> encodings;
    encodings.reserve(_max_tokens);

    for (const auto& encoder : _encoders) {
      auto generated_encodings = encoder->encode(tokens);
      std::cout << "generated encodings size: " << generated_encodings.size()
                << "\n";

      std::copy(generated_encodings.begin(), generated_encodings.end(),
                std::back_inserter(encodings));

      std::cout << "current encoding size: " << encodings.size() << "\n";
      if (encodings.size() > _max_tokens) {
        break;
      }
    }

    return std::vector<uint32_t>(
        encodings.begin(),
        encodings.begin() +
            std::min(static_cast<uint32_t>(encodings.size()), _max_tokens));
  }

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("type", ar::str(type()));
    map->set("_max_tokens", ar::u64(_max_tokens));
    map->set("_sampling_strategy", ar::str(_sampling_strategy));

    throw std::invalid_argument("TO archive has not been implemented yet");
    return map;
  }

  static std::string type() { return "composite"; }

 private:
  friend class cereal::access;

  CompositeEncoder() {}

  uint32_t _max_tokens;
  std::vector<TextEncoderPtr> _encoders;
  std::string _sampling_strategy;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<TextEncoder>(this), _max_tokens, _encoders,
            _sampling_strategy);
  }
};

inline TextEncoderPtr TextEncoder::fromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

  if (type == NGramEncoder::type()) {
    return NGramEncoder::make(archive.u64("n"));
  }
  if (type == PairGramEncoder::type()) {
    return PairGramEncoder::make();
  }
  if (type == FixedDimEncoder::type()) {
    return FixedDimEncoder::make(archive.u64("_max_tokens"));
  }

  throw std::invalid_argument("Invalid encoder type '" + type + "'.");
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::PairGramEncoder)
CEREAL_REGISTER_TYPE(thirdai::dataset::NGramEncoder)
CEREAL_REGISTER_TYPE(thirdai::dataset::FixedDimEncoder)
CEREAL_REGISTER_TYPE(thirdai::dataset::CompositeEncoder)