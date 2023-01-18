#include "StringManipulation.h"

namespace thirdai::utils {

std::vector<std::string_view> splitIntoWords(std::string_view sentence,
                                             char delimiter) {
  std::vector<std::string_view> words;

  bool prev_is_delim = true;
  uint32_t start_of_word_offset;
  for (uint32_t i = 0; i < sentence.size(); i++) {
    if (prev_is_delim && sentence[i] != delimiter) {
      // If we go from a space to a non-space character then we are at the
      // start of a word.
      start_of_word_offset = i;
      prev_is_delim = false;
    }
    if (!prev_is_delim && sentence[i] == delimiter) {
      // If we go from a non-space character to a space then we are at the end
      // of a word.
      uint32_t len = i - start_of_word_offset;

      std::string_view word_view(sentence.data() + start_of_word_offset, len);

      words.push_back(word_view);
      prev_is_delim = true;
    }
  }
  if (!prev_is_delim) {
    // If we don't find a space at the end of the sentence, then there's a
    // last word we need to hash.
    uint32_t len = sentence.size() - start_of_word_offset;

    std::string_view word_view(sentence.data() + start_of_word_offset, len);

    words.push_back(word_view);
  }

  return words;
}

}  // namespace thirdai::utils