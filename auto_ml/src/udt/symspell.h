#include <SymspellCPP/SymSpell.h>
#include <dataset/src/DataSource.h>

class SpellCheckedSentence {
 private:
  std::vector<std::string> tokens;
  std::vector<float> scores;

 public:
  SpellCheckedSentence(const std::vector<std::string>& tokens,
                       const std::vector<float>& scores);

  
  void update_token_and_score(const std::string& token, float score,
                              size_t index, bool inplace = false);

  static int get_score(const SpellCheckedSentence& sentence);

  friend std::ostream& operator<<(std::ostream& os,
                                  const SpellCheckedSentence& sentence);
};

class SymPreTrainer {
 private:
  SymSpell pretrainer;
  int max_edit_distance;
  bool experimental_scores;
  int prefix_length;
  bool use_word_segmentation;

 public:
  static auto make() {
    return std::shared_ptr<SymPreTrainer>(new SymPreTrainer());
  }
  SymPreTrainer(int max_edit_distance, bool experimental_scores = true,
                int prefix_length,
                bool use_word_segmentation);

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
                   std::optional<int>& frequency = std::nullopt);

  void pretrain_file(const dataset::DataSourcePtr& data);
  // Implement the remaining methods of SymPreTrainer as needed
};