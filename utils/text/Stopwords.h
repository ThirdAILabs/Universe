#pragma once

#include <string>
#include <unordered_set>

namespace thirdai::text {

// Set of english stop words as defined by NLTK:
// >>> from nltk.corpus import stopwords
// >>> stop_words = set(stopwords.words('english'))
// With additional stop words added as suggested by this link:
// https://gist.github.com/sebleier/554280?permalink_comment_id=2838826#gistcomment-2838826
extern const std::unordered_set<std::string> stop_words;

}  // namespace thirdai::text
