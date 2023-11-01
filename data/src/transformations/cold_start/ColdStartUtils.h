#pragma once

#include <data/src/ColumnMap.h>

namespace thirdai::data::cold_start {

/**
 * Concatenates each element from the weak phrases with the strong phrase.
 * If strong_sample_num_words is provided, this also independently samples from
 * the strong phrase for every weak phrase.
 */
void mergeStrongWithWeak(std::vector<std::vector<std::string>>& weak_phrases,
                         std::vector<std::string>& strong_phrase,
                         std::optional<uint32_t> strong_sample_num_words,
                         uint32_t seed);

/**
 * Randomly deletes elements from each phrase, resulting in new phrases.
 * Repeats the process num_reps times for each phrase, resulting in (roughly)
 * num_reps * phrases.size() new phrases. Note that if a phrase is not long
 * enough to choose num_to_sample words, then it is kept but only represented
 * once in the output (not num_reps times).
 */
std::vector<std::vector<std::string>> sampleFromPhrases(
    const std::vector<std::vector<std::string>>& phrases,
    uint32_t num_to_sample, uint32_t num_reps, uint32_t seed);

/**
 * Returns a single phrase that takes in the concatenated string of strong
 * columns and returns a strong phrase (this will just be a cleaned version of
 * the input string, possibly length restricted).
 */
std::vector<std::string> getStrongPhrase(
    const std::string& strong_text_in,
    std::optional<uint32_t> max_len = std::nullopt);

/**
 * Creates a phrase by splitting an input string s into whitespace-separated
 * words. Leading and tailing whitespaces are stripped off and ignored.
 */
std::vector<std::string> splitByWhitespace(const std::string& s);

}  // namespace thirdai::data::cold_start