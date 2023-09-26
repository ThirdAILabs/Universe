#include <gtest/gtest.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/symspell.h>
#include <sys/types.h>
#include <ctime>
#include <random>
#include <vector>

using SymPreTrainer = thirdai::automl::udt::SymPreTrainer;
using MapInputBatch = thirdai::dataset::MapInputBatch;

namespace thirdai::symspell::tests {

class SymSpellTest : public testing::Test {
 private:
  automl::udt::SymSpellPtr _symspell_backend;

 public:
  SymSpellTest() {
    _symspell_backend = std::make_shared<SymPreTrainer>(
        SymPreTrainer(automl::udt::defaults::MAX_EDIT_DISTANCE,
                      automl::udt::defaults::PREFIX_LENGTH,
                      automl::udt::defaults::USE_WORD_SEGMENTATION));
  }
  void train(const std::vector<std::string>& inputs) {
    MapInputBatch input_batches;

    for (const auto& str : inputs) {
      MapInput input;

      input["phrase"] = str;
      input_batches.push_back(input);
    }
    std::vector<MapInputBatch> train_data = {input_batches};
    _symspell_backend->pretrain(train_data);
  }

  std::string predict(const std::string& query) {
    // std::pair<std::vector<std::string>, std::vector<float>>
    auto result = _symspell_backend->getCorrectSpellingSingle(query, 1);
    return result.first[0];
  }

  static std::string perturbQuery(const std::string& query, int seed) {
    std::string result = query;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> dist(
        0, query.size() - 1);  // Random index for character replacement

    // Generate a random index
    int rn_id = dist(mt);

    // Generate a random character to replace the character at the random index
    char rn_char = 'a' + mt() % 26;

    // Change the character at the random index
    result[rn_id] = rn_char;

    return result;
  }
};

// Simple test to see if perturbed word is retreived from a set of words
TEST_F(SymSpellTest, SymSpellPredictSingle) {
  std::vector<std::string> train_data = {"apple",  "banana", "cherry", "date",
                                         "fig",    "grape",  "kiwi",   "lemon",
                                         "orange", "pear"};

  train(train_data);
  for (const std::string& query : train_data) {
    const std::string perturbed_query(perturbQuery(query, 42));
    std::string result = predict(perturbed_query);
    ASSERT_EQ(query, result);
  }
}

}  // namespace thirdai::symspell::tests