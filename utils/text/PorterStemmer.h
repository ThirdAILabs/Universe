#pragma once

#include <string>
#include <vector>

namespace thirdai::text::porter_stemmer {

std::string stem(const std::string& word, bool lowercase = true);

std::vector<std::string> stem(const std::vector<std::string>& words,
                              bool lowercase = true);

}  // namespace thirdai::text::porter_stemmer