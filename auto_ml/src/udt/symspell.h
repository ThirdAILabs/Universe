#include <SymspellCPP/include/SymSpell.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <optional>
#include <unordered_map>
#include <regex>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>

class SpellCheckedSentence {
 private:
  std::vector<std::string> tokens;
  std::vector<float> scores;

 public:
  SpellCheckedSentence();

  SpellCheckedSentence(const std::vector<std::string>& tokens,
                       const std::vector<float>& scores);

  SpellCheckedSentence(const SpellCheckedSentence& other);

  SpellCheckedSentence update_token_and_score(const std::string& token, float score,
                              size_t index);

  static float get_score(const SpellCheckedSentence& sentence){
      float total_score = 0.0F;
      for (float score : sentence.scores) {
        total_score += score;
      }
      return total_score;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const SpellCheckedSentence& sentence) {
    for (const std::string& token : sentence.tokens) {
      os << token << " ";
    }
    return os;
  }
};

class SymPreTrainer {
 private:
  SymSpell pretrainer;
  int max_edit_distance;
  bool experimental_scores;
  int prefix_length;
  bool use_word_segmentation;

 public:
  SymPreTrainer();
  
  SymPreTrainer(int max_edit_distance, bool experimental_scores,
                int prefix_length, bool use_word_segmentation);

  std::tuple<std::vector<std::string>, std::vector<float>>
  get_correct_spelling_single(const std::string& word, int top_k = 1);

  std::tuple<std::vector<std::vector<std::string>>,
             std::vector<std::vector<float>>>
  get_correct_spelling_list(const std::vector<std::string>& word_list,
                            int top_k);

  std::vector<SpellCheckedSentence> correct_sentence(
      std::vector<std::string> tokens_list, int predictions_per_token,
      int maximum_candidates, bool stop_if_found);

  void index_words(std::vector<std::string> words_to_index,
                   std::vector<int> frequency);

  void pretrain_file(const thirdai::dataset::DataSourcePtr& data, std::string correct_column_name);

};