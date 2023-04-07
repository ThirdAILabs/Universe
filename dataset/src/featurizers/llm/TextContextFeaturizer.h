#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::dataset {

class TextContextFeaturizer {
 public:
  TextContextFeaturizer(uint32_t lrc_len, uint32_t irc_len, uint32_t src_len,
                        uint32_t vocab_size)
      : _lrc_len(lrc_len),
        _irc_len(irc_len),
        _src_len(src_len),
        _vocab_size(vocab_size) {}

  TextContextFeaturizer() {}

  BoltVector lrcContext(const std::vector<uint32_t>& tokens) const {
    return lrcContext(tokens, tokens.size());
  }

  BoltVector lrcContext(const std::vector<uint32_t>& tokens,
                        uint32_t label_index) const;

  BoltVector ircContext(const std::vector<uint32_t>& tokens) const {
    return ircContext(tokens, tokens.size());
  }

  BoltVector ircContext(const std::vector<uint32_t>& tokens,
                        uint32_t label_index) const;

  BoltVector srcContext(const std::vector<uint32_t>& tokens) const {
    return srcContext(tokens, tokens.size());
  }

  BoltVector srcContext(const std::vector<uint32_t>& tokens,
                        uint32_t label_index) const;

 private:
  uint32_t _lrc_len;
  uint32_t _irc_len;
  uint32_t _src_len;
  uint32_t _vocab_size;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_lrc_len, _irc_len, _src_len, _vocab_size);
  }
};

}  // namespace thirdai::dataset