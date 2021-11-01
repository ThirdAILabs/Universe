#include "../../../dataset/string/loaders/SentenceLoader.h"
#include <gtest/gtest.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * 1. Sentence loader
 *  a. Correct fragmentation:
 *      - Start with a vector of sentences that only has lowercase letters,
 spaces and
 *      numbers. No punctuation. Stream each sentence to a file with a full stop
 at
 *      the end. Queue this file into the string loader. We should get a vector
 of strings
 *      resembling the vector we started with.
 *  b. Correct preprocessing:
 *      - Given a text file that has uppercase letters, whitespaces other than
 space, and
 *      other symbols, make sure that:
 *          - the output strings only have spaces, numbers, and lowercase
 characters.
 *          - the output strings do not have consecutive spaces.
 *          - the number of output strings equals the number of expected
 sentences.
 *  c. Correct batch size:
 *      - Given a target batch size, the size of the vector of loaded strings is
 * at most equal to the target batch size.
 *      - Return value of loadStringsAndLabels is equal to size of vector of
 loaded strings
 *      - If file is exhausted, loadedStringsAndLabels returns 0 and vector of
 loaded strings
 * is empty
 *  d. Correct labels:
 *      - The vector of loaded strings and the vector of loaded labels are equal
 in size.
 *      - Each entry in the vector of loaded labels is an empty vector.
 */

using thirdai::utils::dataset::SentenceLoader;

static std::string lorem_ipsum_passage =
    "Lorem Ipsum is simply dummy text of the printing and typesetting "
    "industry. Lorem Ipsum has been the industry's standard dummy text ever "
    "since the 1500s, when an unknown printer took a galley of type and "
    "scrambled it to make a type specimen book. It has survived not only five "
    "centuries, but also the leap into electronic typesetting, remaining "
    "essentially unchanged. It was popularised in the 1960s with the release "
    "of Letraset sheets containing Lorem Ipsum passages, and more recently "
    "with desktop publishing software like Aldus PageMaker including versions "
    "of Lorem Ipsum.\n\nWhy do we use it?\nIt is a long established fact that "
    "a reader will be distracted by the readable content of a page when "
    "looking at its layout. The point of using Lorem Ipsum is that it has a "
    "more-or-less normal distribution of letters, as opposed to using 'Content "
    "here, content here', making it look like readable English. Many desktop "
    "publishing packages and web page editors now use Lorem Ipsum as their "
    "default model text, and a search for 'lorem ipsum' will uncover many web "
    "sites still in their infancy. Various versions have evolved over the "
    "years, sometimes by accident, sometimes on purpose (injected humour and "
    "the like). \n Where does it come from?\nContrary to popular belief, Lorem "
    "Ipsum is not simply random text. It has roots in a piece of classical "
    "Latin literature from 45 BC, making it over 2000 years old. Richard "
    "McClintock, a Latin professor at Hampden-Sydney College in Virginia, "
    "looked up one of the more obscure Latin words, consectetur, from a Lorem "
    "Ipsum passage, and going through the cites of the word in classical "
    "literature, discovered the undoubtable source. Lorem Ipsum comes from "
    "sections 1.10.32 and 1.10.33 of \"de Finibus Bonorum et Malorum\" (The "
    "Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a "
    "treatise on the theory of ethics, very popular during the Renaissance. "
    "The first line of Lorem Ipsum, \"Lorem ipsum dolor sit amet..\", comes "
    "from a line in section 1.10.32.\nThe standard chunk of Lorem Ipsum used "
    "since the 1500s is reproduced below for those interested. Sections "
    "1.10.32 and 1.10.33 from \"de Finibus Bonorum et Malorum\" by Cicero are "
    "also reproduced in their exact original form, accompanied by English "
    "versions from the 1914 translation by H. Rackham.";

static void checkOnlyWantedCharacters(std::string& str) {
  for (auto& c : str) {
    if (!(c == ' ' || ('0' <= c && c <= '9') || ('a' <= c && c <= 'z'))) {
      std::cout << "The string " << str << " contains the character '" << c
                << "'" << std::endl;
    }
    ASSERT_TRUE(c == ' ' || ('0' <= c && c <= '9') || ('a' <= c && c <= 'z'));
  }
}

static void checkNoConsecutiveSpaces(std::string& str) {
  char last_c = 'a';
  for (auto& c : str) {
    if (c == ' ' && last_c == ' ') {
      std::cout << "There are consecutive spaces in the string '" << str << "'"
                << std::endl;
    }
    ASSERT_FALSE(c == ' ' && last_c == ' ');
    last_c = c;
  }
}

static void checkStringNotEmpty(std::string& str) {
  if (str.empty()) {
    std::cout << "Found an empty string!" << std::endl;
  }
  ASSERT_FALSE(str.empty());
}

static void generateTwoFilesWithOneDifferentSentence(
    std::string& filename1, std::string& filename2, std::string& file1sentence,
    std::string& file2sentence, std::string& shared_string) {
  std::ofstream out_file(filename1, std::ios::trunc);
  out_file << file1sentence << "." << std::endl;
  out_file << shared_string << std::endl;
  out_file.close();
  out_file.open(filename2, std::ios::trunc);
  out_file << file2sentence << "." << std::endl;
  out_file << shared_string << std::endl;
  out_file.close();
}

static void checkSentenceCountsOfTwoFilesWithOneDifferentSentence(
    std::unordered_map<std::string, uint32_t> counts,
    std::string& file1sentence, std::string& file2sentence) {
  // Since the files contain "{filename1}." and "{filename2}."" respectively,
  // filename1 and filename2 should each appear once while other strings must
  // appear an even number of times.
  ASSERT_EQ(counts[file1sentence], 1);
  ASSERT_EQ(counts[file2sentence], 1);
  for (const auto& kv : counts) {
    if (kv.first != file1sentence && kv.first != file2sentence) {
      ASSERT_TRUE(kv.second % 2 == 0);
    }
  }
}

static void generateSimpleFileFromVectorOfStrings(
    std::vector<std::string>& strings, std::string& filename) {
  std::ofstream out_file(filename);
  for (std::string& str : strings) {
    for (auto& c : str) {
      if (!(c == ' ' || ('0' <= c && c <= '9') || ('a' <= c && c <= 'z'))) {
        std::stringstream ss;
        ss << "The vector of strings has unwanted characters. Found '" << c
           << "'";
        throw std::invalid_argument(ss.str());
      }
    }
    char last_c = 'a';
    for (auto& c : str) {
      if (c == ' ' && last_c == ' ') {
        throw std::invalid_argument(
            "The vector of strings has a sentence with consecutive spaces: '" +
            str + "'");
      }
      last_c = c;
    }
    out_file << str << ". ";
  }
  out_file.close();
}

/**
 *  a. Correct fragmentation:
 *      - Start with a vector of sentences that only has lowercase letters,
 * spaces and numbers. No punctuation. Stream each sentence to a file with a
 * full stop at the end. Queue this file into the string loader. We should get a
 * vector of strings resembling the vector we started with.
 */
TEST(SentenceLoaderTest, CorrectlyFragmentsSimpleFile) {
  std::vector<std::string> strings;
  strings.push_back("hello there");
  strings.push_back("this is the first test");
  strings.push_back("this is a very simple test");
  strings.push_back(
      "as you can see the input strings have the same format as the expected "
      "output strings");
  strings.push_back(
      "we stream these strings into a file and delimit them with full stops "
      "and spaces");
  strings.push_back(
      "if the final result is the same as these strings then the file is "
      "correctly fragmented");
  std::string filename = "simple_sentences.txt";
  generateSimpleFileFromVectorOfStrings(strings, filename);

  SentenceLoader loader;
  std::ifstream file(filename);
  std::vector<std::string> loaded_strings;
  std::vector<std::vector<uint32_t>> loaded_labels;
  size_t i = 0;
  while (loader.loadStringsAndLabels(file, 1, loaded_strings, loaded_labels) ==
         1) {
    ASSERT_EQ(loaded_strings[0], strings[i]);
    i++;
  }
  ASSERT_TRUE(i > 0);
  ASSERT_EQ(i, strings.size());
}

/**
 *  b. Correct preprocessing:
 *      - Given a text file that has uppercase letters, whitespaces other than
 * space, and other symbols, make sure that:
 *          - the output strings only have spaces, numbers, and lowercase
 * characters.
 *          - the output strings do not have consecutive spaces.
 *      - Note that this text file does not have typos or adversary strings that
 * would make it tricky to parse, such as consecutive spaces and ending
 * punctuations like ". ?!? ." If confronted with a tricky dataset as the above,
 * all unwanted symbols may be removed, but there can be unwanted scenarios of
 * consecutive spaces or sentences consisting only of spaces.
 */
TEST(SentenceLoaderTest, CorrectPreprocessing) {
  std::string out_file_name = "lorem_ipsum_test.txt";
  std::ofstream out_file(out_file_name, std::ios::trunc);
  out_file << lorem_ipsum_passage;
  out_file.close();
  SentenceLoader loader;
  std::ifstream out_file_read(out_file_name);
  std::vector<std::string> loaded_strings;
  std::vector<std::vector<uint32_t>> loaded_labels;
  size_t loaded_count = 0;
  while (loader.loadStringsAndLabels(out_file_read, 1, loaded_strings,
                                     loaded_labels)) {
    loaded_count++;
    checkOnlyWantedCharacters(loaded_strings[0]);
    checkNoConsecutiveSpaces(loaded_strings[0]);
    checkStringNotEmpty(loaded_strings[0]);
  }
  ASSERT_TRUE(loaded_count > 0);
}

/**
 *  c. Correct batch size:
 *      - Given a target batch size, the size of the vector of loaded strings is
 * at most equal to the target batch size.
 *      - Return value of loadStringsAndLabels is equal to size of vector of
 * loaded strings
 *      - If file is exhausted, loadedStringsAndLabels returns 0 and vector of
 * loaded strings is empty
 */
TEST(SentenceLoaderTest, CorrectBatchSize) {
  std::string out_file_name = "lorem_ipsum_test.txt";
  std::ofstream out_file(out_file_name, std::ios::trunc);
  out_file << lorem_ipsum_passage;
  out_file.close();
  SentenceLoader loader;
  std::ifstream out_file_read(out_file_name);
  std::vector<std::string> loaded_strings;
  std::vector<std::vector<uint32_t>> loaded_labels;
  size_t loaded_count = 0;
  uint32_t target_batch_size = 10;
  uint32_t loaded_strings_count;
  uint32_t times_loaded_less_than_target_batch_size = 0;
  while ((loaded_strings_count =
              loader.loadStringsAndLabels(out_file_read, target_batch_size,
                                          loaded_strings, loaded_labels)) > 0) {
    ASSERT_EQ(loaded_strings_count, loaded_strings.size());
    ASSERT_LE(loaded_strings_count, target_batch_size);
    if (loaded_strings_count < target_batch_size) {
      ASSERT_EQ(times_loaded_less_than_target_batch_size,
                0);  // Cannot load > 0 but < target_batch_size strings more
                     // than once.
      times_loaded_less_than_target_batch_size++;
    }
    loaded_count++;
  }
  ASSERT_EQ(loaded_strings.size(), 0);
  // File is exhausted; next load count should be zero again.
  ASSERT_EQ(loader.loadStringsAndLabels(out_file_read, target_batch_size,
                                        loaded_strings, loaded_labels),
            0);
  ASSERT_TRUE(loaded_count > 0);
}

/**
 *  d. Correct labels:
 *      - The vector of loaded strings and the vector of loaded labels are equal
 * in size.
 *      - Each entry in the vector of loaded labels is an empty vector.
 */
TEST(SentenceLoaderTest, CorrectLabels) {
  std::string out_file_name = "lorem_ipsum_test.txt";
  std::ofstream out_file(out_file_name, std::ios::trunc);
  out_file << lorem_ipsum_passage;
  out_file.close();
  SentenceLoader loader;
  std::ifstream out_file_read(out_file_name);
  std::vector<std::string> loaded_strings;
  std::vector<std::vector<uint32_t>> loaded_labels;
  uint32_t target_batch_size = 10;
  while (loader.loadStringsAndLabels(out_file_read, target_batch_size,
                                     loaded_strings, loaded_labels) > 0) {
    ASSERT_EQ(loaded_strings.size(), loaded_labels.size());
    for (const auto& labels_vec : loaded_labels) {
      ASSERT_EQ(labels_vec.size(), 0);  // SentenceLoader does not load labels.
    }
  }
}