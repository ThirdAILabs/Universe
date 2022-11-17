#pragma once

#include <new_dataset/src/featurization_pipeline/Augmentation.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <memory>
#include <string>
#include <vector>

namespace thirdai::dataset {

/**
 * This class augments text data by applying various slicing and sampling
 * methods to sequences of words. It takes in a ColumnMap with text columns
 * and a label column, and it outputs a ColumnMap with a label column and
 * a single text column. The output label column has the same name as the input
 * and the text column has a name specified by the user. The text columns are
 * are separated into "strong signal" columns that carry a strong signal (title
 * keywords, etc) and "weak signal" columns that carry a supplementary signal
 * (description, reviews etc). The weak columns are concatenated and sliced 
 * into "natural" phrases and "chunk" phrases. Natural phrases are those that 
 * are delimited by some form of punctuation while chunked phrases are simply 
 * split by a pre-specified word count. The strong columns are also
 * concatenated but are split by whitespace (not punctuation) to ensure that
 * words from the strong column are present in every row of the output data.
 * The strong and weak phrases are optionally sub-sampled to create more data
 * and then written to a new ColumnMap with the corresponding labels.
 */
class ColdStartTextAugmentation final : public Augmentation {
 public:
  ColdStartTextAugmentation(
    std::vector<std::string> strong_column_names,
    std::vector<std::string> weak_column_names,
    std::string label_column_name,
    std::string output_column_name,
    std::optional<uint32_t> weak_min_len = std::nullopt,
    std::optional<uint32_t> weak_max_len = std::nullopt,
    std::optional<uint32_t> weak_chunk_len = std::nullopt,
    std::optional<uint32_t> weak_downsample_num = std::nullopt,
    uint32_t weak_downsample_reps = 1,
    std::optional<uint32_t> strong_max_len = std::nullopt,
    std::optional<uint32_t> strong_downsample_num = std::nullopt,
    uint32_t seed = 42803)
      : _strong_column_names(std::move(strong_column_names)),
        _weak_column_names(std::move(weak_column_names)),
        _label_column_name(std::move(label_column_name)),
        _output_column_name(std::move(output_column_name)),
        _weak_min_len(weak_min_len),
        _weak_max_len(weak_max_len),
        _weak_chunk_len(weak_chunk_len),
        _weak_downsample_num(weak_downsample_num),
        _weak_downsample_reps(weak_downsample_reps),
        _strong_max_len(strong_max_len),
        _strong_downsample_num(strong_downsample_num),
        _seed(seed) {}
    // Constructs a data augmentation process for strong and weak text columns.
    // 
    // Arguments:
    //    strong_column_names: A list of columns containing strong signal, such
    //        as titles or keywords.
    //    weak_column_names: A list of columns containing a weak or
    //        supplementary signal, such as product reviews or descriptions.
    //    label_column_name: Contains multi-class labels for each text.
    //    output_column_name: The name of the column containing augmented text.
    //    weak_min_len: If provided, then natural phrases are encouraged to be
    //        at least this many words. Note that some inputs may generate
    //        smaller natural phrases even if this parameter is provided, if
    //        there are not enough words in the weak text or the last natural
    //        phrase in the text is not long enough.
    //    weak_max_len: If provied, then natural phrases are forced to be
    //        smaller than this many words.
    //    weak_chunk_len: If provided, then we also include "chunked phrases"
    //        of the specified length. These are included in addition to the
    //        natural phrases, as they are helpful (but not sufficient) for
    //        good performance. If not provided, then chunked phrases are
    //        not included.
    //    weak_downsample_num: If provided, then each weak phrase is 
    //        sub-sampled to the specified number of words (if possible - there
    //        may not be enough words for short phrases).
    //    weak_downsample_reps: This determines the number of times the
    //        sampling process is repeated for each weak phrase. Note that
    //        this directly multiplies the number of weak phrases; therefore,
    //        we do not suggest values larger than 10 due to large output size.
    //    strong_max_len: If provided, then the concatenated strong phrase is
    //        cut off after the specified number of words.
    //    strong_downsample_num: If provided, then the strong phrase is
    //        independently sub-sampled to the specified number of words before
    //        being concatenated with each phrase (if possible - there may not
    //        be enough words for short phrases).
    //    seed: Seed for the random number generator.

  ColumnMap apply(const ColumnMap& columns) final;

 private:
  typedef std::vector<std::string> Phrase_t;
  typedef std::vector<Phrase_t> PhraseCollection_t;

  std::vector<std::string> _strong_column_names;
  std::vector<std::string> _weak_column_names;
  std::string _label_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _weak_min_len;
  std::optional<uint32_t> _weak_max_len;
  std::optional<uint32_t> _weak_chunk_len;
  std::optional<uint32_t> _weak_downsample_num;
  uint32_t _weak_downsample_reps;
  std::optional<uint32_t> _strong_max_len;
  std::optional<uint32_t> _strong_downsample_num;
  uint32_t _seed;

  Phrase_t splitByWhitespace(std::string& s);
  // Creates a phrase by splitting an input string s into whitespace-separated
  // words. Leading and tailing whitespaces are stripped off and ignored.

  void replacePunctuation(std::string& s);
  // Replaces punctuation characters in s with whitespace.

  void stripWhitespace(std::string& s);
  // Strips leading and tailing whitespace.

  PhraseCollection_t getPhrases(std::string& s);
  // Returns a set of natural and chunked phrases from s.

  void sampleFromPhrases(PhraseCollection_t &phrases,
    uint32_t num_to_sample, uint32_t num_reps);
  // Randomly deletes elements from each phrase, resulting in new phrases.
  // Repeats the process num_reps times for each phrase, resulting in
  // N*num_reps phrases if there were originally N phrases.

  void mergeStrongWithWeak(PhraseCollection_t &weak_phrases,
    Phrase_t &strong_phrase);
  // Concatenates each element from the weak phrases with the strong phrase.
  // If _strong_downsample_num was provided in the constructor, this also
  // independently samples from the strong phrase for every weak phrase.

  // Private constructor for cereal.
  ColdStartTextAugmentation()
      : _strong_column_names(),
        _weak_column_names(),
        _label_column_name(),
        _output_column_name(),
        _weak_min_len(0),
        _weak_max_len(0),
        _weak_chunk_len(0),
        _weak_downsample_num(0),
        _weak_downsample_reps(1),
        _strong_max_len(0),
        _strong_downsample_num(0),
        _seed(42803){}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Augmentation>(this), _strong_column_names,
            _weak_column_names, _label_column_name, _output_column_name,
            _weak_min_len, _weak_max_len, _weak_chunk_len,
            _weak_downsample_num, _strong_max_len, _strong_downsample_num);
  }

};

}  // namespace thirdai::dataset