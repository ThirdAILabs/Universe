#pragma once

#include <string>
#include <unordered_set>
#include <vector>
namespace thirdai::data::ner::utils {

using TokenTags = std::vector<std::pair<std::string, float>>;
using SentenceTags = std::vector<TokenTags>;

bool isNumberWithPunct(const std::string& s,
                       const std::unordered_set<char>& exception_chars);

bool containsAlphabets(const std::string& input,
                       const std::unordered_set<char>& exception_chars);

bool containsNumbers(const std::string& input,
                     const std::unordered_set<char>& exception_chars);

std::string trimPunctuation(const std::string& str);

std::vector<std::string> cleanAndLowerCase(
    const std::vector<std::string>& tokens);

std::string stripNonDigits(const std::string& input);

bool containsKeywordInRange(const std::vector<std::string>& tokens,
                            size_t start, size_t end,
                            const std::unordered_set<std::string>& keywords);

bool luhnCheck(const std::string& number);

std::string findContiguousNumbers(const std::vector<std::string>& v,
                                  uint32_t index, uint32_t k = 3);

uint32_t find_max_contiguous_window(const SentenceTags& sentence_tags,
                                    uint32_t index,
                                    const std::string& tag_to_find);
}  // namespace thirdai::data::ner::utils