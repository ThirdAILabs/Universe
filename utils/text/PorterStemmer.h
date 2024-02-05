#pragma once

#include <string>
#include <vector>

namespace thirdai::text::porter_stemmer {

/**
 * Based on the Porter stemming algorithm proposed in:
 *    Porter, M. "An algorithm for suffix stripping."
 *        Program 14.3 (1980): 130-137.
 *
 * Overview of algorithm: https://tartarus.org/martin/PorterStemmer/def.txt
 */
std::string stem(const std::string& word);

std::vector<std::string> stem(const std::vector<std::string>& words);

}  // namespace thirdai::text::porter_stemmer