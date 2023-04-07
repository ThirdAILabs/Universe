#include "TextContextFeaturizer.h"
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::dataset {

BoltVector TextContextFeaturizer::lrcContext(
    const std::vector<uint32_t>& tokens, uint32_t label_index) const {
  uint32_t lrc_len = std::min(label_index, _lrc_len);

  const uint32_t* context_start = tokens.data() + label_index - lrc_len;

  BoltVector vector(/* l= */ lrc_len, /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(context_start, context_start + lrc_len, vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

BoltVector TextContextFeaturizer::ircContext(
    const std::vector<uint32_t>& tokens, uint32_t label_index) const {
  uint32_t irc_len = std::min(label_index, _irc_len);

  std::vector<uint32_t> irc_context =
      token_encoding::unigramPreservingPairgrams(
          tokens.data() + label_index - irc_len, irc_len, _vocab_size);

  BoltVector vector(/* l= */ irc_context.size(), /* is_dense= */ false,
                    /* has_gradient= */ false);
  std::copy(irc_context.begin(), irc_context.end(), vector.active_neurons);
  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

BoltVector TextContextFeaturizer::srcContext(
    const std::vector<uint32_t>& tokens, uint32_t label_index) const {
  uint32_t src_len = std::min(label_index, _src_len);
  uint32_t padding_len = _src_len - src_len;

  const uint32_t* context_start = tokens.data() + label_index - src_len;

  BoltVector vector(/* l= */ _src_len, /* is_dense= */ false,
                    /* has_gradient= */ false);

  // Zero pad if short range context length is greater than number of tokens. We
  // pad the begining so that the last token before the prediction is always at
  // the end.
  std::fill_n(vector.active_neurons, padding_len, 0);
  std::copy(context_start, context_start + src_len,
            vector.active_neurons + padding_len);
  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

}  // namespace thirdai::dataset