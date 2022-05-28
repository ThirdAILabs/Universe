#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <functional>
#include <type_traits>

namespace thirdai::dataset {

struct TextEncodingUtils {
  /**
   * Deduplicates indices by summing values and adds features to the given
   * vector. All indices expected to correspond to the same value.
   */
  inline static void sumRepeatedIndices(std::vector<uint32_t>& indices,
                                        float value, ExtendableVector& vec) {
    // Put equivalent indices next to each other.
    std::sort(indices.begin(), indices.end());

    /**
     * If current index is the same as the next index, keep accumulating
     * val. Otherwise, add sparse feature at the current index with the
     * accumulated value and reset val.
     */
    float val = 0.0;
    uint32_t i = 0;
    for (; i < indices.size() - 1; ++i) {
      uint32_t idx = indices[i];
      uint32_t next_idx = indices[i + 1];
      val += value;

      if (idx != next_idx) {
        vec.addExtensionSparseFeature(idx, val);
        val = 0.0;  // Reset val since next idx is different.
      }
    }

    /**
     * If we're looking at the last element, the next element is clearly
     * "different", so we add a sparse feature accordingly.
     */
    if (i == indices.size() - 1) {
      val += value;
      vec.addExtensionSparseFeature(indices.back(), val);
    }
  }

  /**
   * Parses through a sentence and does something to each word.
   */
  template <typename WORD_PROCESSOR_T>
  inline static void forEachWord(std::string& sentence,
                                 WORD_PROCESSOR_T word_processor) {
    static_assert(
        std::is_convertible<WORD_PROCESSOR_T,
                            std::function<void(char*, size_t)>>::value);

    const auto* start_ptr = sentence.c_str();
    bool last_ptr_was_space = false;
    for (const auto& c : sentence) {
      // Just saw a word boundary
      if (isspace(c) && !last_ptr_was_space) {
        last_ptr_was_space = true;
        size_t len = std::distance(start_ptr, &c);
        word_processor(start_ptr, len);
      }

      // Encountered a new word
      if (last_ptr_was_space && !isspace(c)) {
        last_ptr_was_space = false;
        start_ptr = &c;
      }
    }

    // Don't leave out last word.
    if (!last_ptr_was_space) {
      size_t cur_pos = std::distance(sentence.c_str(), start_ptr);
      uint32_t len = sentence.size() - cur_pos;
      word_processor(start_ptr, len);
    }
  }

  /**
   * Creates a copy of the original string where all characters are lowercase.
   */
  inline static std::string makeLowerCase(const std::string& original) {
    std::string lower_case_text = original;
    for (auto& c : lower_case_text) {
      c = std::tolower(c);
    }
    return lower_case_text;
  }
};

}  // namespace thirdai::dataset