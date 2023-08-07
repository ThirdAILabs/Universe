#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::dataset {

class TextContextFeaturizer {
 public:
  TextContextFeaturizer(uint32_t lrc_len, uint32_t irc_len, uint32_t src_len,
                        uint32_t vocab_size, bool need_position_context = false)
      : _lrc_len(lrc_len),
        _irc_len(irc_len),
        _src_len(src_len),
        _vocab_size(vocab_size),
        _needs_position_context(need_position_context) {}

  TextContextFeaturizer() {}

  BoltVector lrcContext(const std::vector<uint32_t>& tokens) const {
    return lrcContext(tokens, tokens.size(), 0);
  }

  BoltVector lrcContext(const std::vector<uint32_t>& tokens,
                        uint32_t end_index) const {
    return lrcContext(tokens, end_index, 0);
  }

  // Returns up to the last lrc_len tokens before end_index represented as
  // unigrams.
  BoltVector lrcContext(const std::vector<uint32_t>& tokens, uint32_t end_index,
                        uint32_t start_index) const;

  BoltVector ircContext(const std::vector<uint32_t>& tokens) const {
    return ircContext(tokens, tokens.size(), 0);
  }

  BoltVector ircContext(const std::vector<uint32_t>& tokens,
                        uint32_t end_index) const {
    return ircContext(tokens, end_index, 0);
  }

  // Returns up to the last irc_len tokens before end_index represented as
  // pairgrams.
  BoltVector ircContext(const std::vector<uint32_t>& tokens, uint32_t end_index,
                        uint32_t start_index) const;

  BoltVector srcContext(const std::vector<uint32_t>& tokens) const {
    return srcContext(tokens, tokens.size(), 0);
  }
  BoltVector srcContext(const std::vector<uint32_t>& tokens,
                        uint32_t end_index) const {
    return srcContext(tokens, end_index, 0);
  }

  // Returns up to the last src_len tokens before end_index represented as
  // unigrams.
  BoltVector srcContext(const std::vector<uint32_t>& tokens, uint32_t end_index,
                        uint32_t start_index) const;

  BoltVector positionContext(const std::vector<uint32_t>& tokens) const {
    return positionContext(tokens, tokens.size(), 0);
  }

  BoltVector positionContext(const std::vector<uint32_t>& tokens,
                             uint32_t end_index) const {
    return positionContext(tokens, end_index, 0);
  }

  // Returns the positional context before end_index represented as
  // unigrams.
  BoltVector positionContext(const std::vector<uint32_t>& tokens,
                             uint32_t end_index, uint32_t start_index) const;

  uint32_t getLongRangeContextLength() const { return _lrc_len; }

  bool needPositionContext() const { return _needs_position_context; }

 private:
  uint32_t _lrc_len;
  uint32_t _irc_len;
  uint32_t _src_len;
  uint32_t _vocab_size;
  bool _needs_position_context;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_lrc_len, _irc_len, _src_len, _vocab_size, _needs_position_context);
  }
};

}  // namespace thirdai::dataset