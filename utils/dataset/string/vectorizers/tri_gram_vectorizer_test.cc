#include "TriGramVectorizer.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

using thirdai::utils::TriGramVectorizer;

class TriGramVectorizerTest : public testing::Test {
 private:
  static std::string generate_random_string() {
    const uint32_t num_chars = 26;
    const uint32_t starting_ascii = 65;
    std::string str = "AAAAAA";
    str[0] = rand() % num_chars + starting_ascii;
    str[1] = rand() % num_chars + starting_ascii;
    str[2] = rand() % num_chars + starting_ascii;

    str[3] = rand() % num_chars + starting_ascii;
    str[4] = rand() % num_chars + starting_ascii;
    str[5] = rand() % num_chars + starting_ascii;
    return str;
  }

  static std::vector<std::string> generate_all_trigrams() {
    uint8_t chars[37];
    // Space
    chars[0] = 32;
    // Numbers
    for (size_t i = 0; i < 10; i++) {
      chars[1 + i] = 48 + i;
    }
    // Lower case letters
    for (size_t i = 0; i < 26; i++) {
      chars[11 + i] = 97 + i;
    }
    std::vector<std::string> trigrams;
    trigrams.reserve(50653);
    for (size_t i = 0; i < 37; i++) {
      for (size_t j = 0; j < 37; j++) {
        for (size_t k = 0; k < 37; k++) {
          std::string str = "aaa";
          str[0] = chars[i];
          str[1] = chars[j];
          str[2] = chars[k];
          trigrams.push_back(str);
        }
      }
    }
    return trigrams;
  }

  //  protected:
  // Issues: I don't know how to get rid of these errors. I should try working
  // with the cmakelists. IT WORKSSS For now I can also just finish the generate
  // trigrams function. DONE Or I can focus on writing the actual test cases
  // first <- Now this is all I have to do. I kinda wanna map out the bigger
  // picture first though.
};
