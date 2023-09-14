#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::dataset {

class TextContextFeaturizer {
 public:
  TextContextFeaturizer(uint32_t lrc_len, uint32_t irc_len, uint32_t src_len,
                        uint32_t vocab_size, bool include_position = false)
      : _lrc_len(lrc_len),
        _irc_len(irc_len),
        _src_len(src_len),
        _vocab_size(vocab_size),
        _include_position(include_position) {}

  TextContextFeaturizer() {}

  BoltVector lrcContext(const std::vector<uint32_t>& tokens) const {
    return lrcContext(tokens, 0, tokens.size());
  }

  // Returns up to the last lrc_len tokens before end_index represented as
  // unigrams.
  BoltVector lrcContext(const std::vector<uint32_t>& tokens,
                        uint32_t start_index, uint32_t end_index) const;

  BoltVector ircContext(const std::vector<uint32_t>& tokens) const {
    return ircContext(tokens, 0, tokens.size());
  }

  // Returns up to the last irc_len tokens before end_index represented as
  // pairgrams.
  BoltVector ircContext(const std::vector<uint32_t>& tokens,
                        uint32_t start_index, uint32_t end_index) const;

  BoltVector srcContext(const std::vector<uint32_t>& tokens) const {
    return srcContext(tokens, 0, tokens.size());
  }

  // Returns up to the last src_len tokens before end_index represented as
  // unigrams.
  BoltVector srcContext(const std::vector<uint32_t>& tokens,
                        uint32_t start_index, uint32_t end_index) const;

  size_t contextSize() const { return _lrc_len; }

  uint32_t vocabSize() const { return _vocab_size; }

 private:
  uint32_t _lrc_len;
  uint32_t _irc_len;
  uint32_t _src_len;
  uint32_t _vocab_size;
  bool _include_position;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_lrc_len, _irc_len, _src_len, _vocab_size, _include_position);
  }
};

}  // namespace thirdai::dataset