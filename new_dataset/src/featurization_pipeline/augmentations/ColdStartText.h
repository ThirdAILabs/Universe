#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
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
 * and the text column has a name specified by the user. The input text columns
 * are separated into "strong signal" columns that carry a strong signal (title
 * keywords, etc) and "weak signal" columns that carry a supplementary signal
 * (description, reviews etc). The augmented data consists of combined words
 * from strong and weak columns. The weak columns are concatenated and sliced 
 * into "natural" phrases and "chunk" phrases. Natural phrases are those that 
 * are delimited by some form of punctuation while chunked phrases consist of 
 * sequences of a pre-specified number of consecutive words. The strong columns
 * are also concatenated but are split by whitespace (not punctuation) into a
 * single phrase that is present in every row of the output data. The strong
 * and weak phrases are also optionally sub-sampled to create more data
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
    std::optional<uint32_t> weak_sample_num_words = std::nullopt,
    uint32_t weak_sample_reps = 1,
    std::optional<uint32_t> strong_max_len = std::nullopt,
    std::optional<uint32_t> strong_sample_num_words = std::nullopt,
    uint32_t seed = 42803);
    /*
    Constructs a data augmentation process for strong and weak text columns.
    
    Arguments:
       strong_column_names: A list of columns containing strong signal, such
           as titles or keywords.
       weak_column_names: A list of columns containing a weak or
           supplementary signal, such as product reviews or descriptions.
       label_column_name: Contains multi-class labels for each text.
       output_column_name: The name of the column containing augmented text.
       weak_min_len: If provided, then natural phrases are encouraged to be
           at least this many words. This is done by concatenating consecutive
           natural phrases. For example, 'red shirt, sturdy wool material' has 
           two natural phrases: 'red shirt' and 'sturdy wool material', but it
           will be merged into 'red shirt sturdy wool material' if weak_min_len
           is smaller than 2. Note that some inputs may generate smaller
           natural phrases even if this parameter is provided, if there are not
           enough words in the weak text or the final natural phrase in the
           text is not long enough. 
       weak_max_len: If provided, then natural phrases are forced to be
           smaller than this many words. This is done by cutting the phrase
           before a natural delimiter once we have a phrase of weak_max_len
           words.
       weak_chunk_len: If provided, then we also include "chunked phrases" -
           sequences of consecutive words that are obtained by splitting the
           text into "chunks" of the specified length, without regard for
           punctuation or other natural delimiters. These are included in
           addition to the natural phrases, as they are helpful (but not
           sufficient) for good performance. If not provided, then chunked
           phrases are not included.
       weak_sample_num_words: If provided, then each weak phrase is 
           sub-sampled to the specified number of words (if possible - there
           may not be enough words for short phrases). This operation is
           applied to the phrases that are extracted via chunking / natural
           phrase delimiters and acts as a dropout regularizer as well as a way
           to standardize the number of words in each phrase (as natural
           phrases can vary in size).
       weak_sample_reps: This determines the number of times the
           sampling process is repeated for each weak phrase. Note that
           this approximately multiplies the number of weak phrases; therefore,
           we do not suggest values larger than 10 due to large output size.
       strong_max_len: If provided, then the concatenated strong phrase is
           cut off after the specified number of words.
       strong_sample_num_words: If provided, then the strong phrase is
           independently sub-sampled to the specified number of words before
           being concatenated with each phrase (if possible - there may not
           be enough words for short phrases).
       seed: Seed for the random number generator.
    */

  ColumnMap apply(const ColumnMap& columns) final;

 private:
  typedef std::vector<std::string> Phrase;
  typedef std::vector<Phrase> PhraseCollection;

  std::vector<std::string> _strong_column_names;
  std::vector<std::string> _weak_column_names;
  std::string _label_column_name;
  std::string _output_column_name;
  std::optional<uint32_t> _weak_min_len;
  std::optional<uint32_t> _weak_max_len;
  std::optional<uint32_t> _weak_chunk_len;
  std::optional<uint32_t> _weak_sample_num_words;
  uint32_t _weak_sample_reps;
  std::optional<uint32_t> _strong_max_len;
  std::optional<uint32_t> _strong_sample_num_words;
  uint32_t _seed;

  static Phrase splitByWhitespace(std::string& s);
  /*
  Creates a phrase by splitting an input string s into whitespace-separated
  words. Leading and tailing whitespaces are stripped off and ignored.
  */

  static void replacePunctuationWithSpaces(std::string& s);
  /*
  Replaces punctuation characters in s with whitespace.
  */

  static void stripWhitespace(std::string& s);
  /*
  Strips leading and tailing whitespace.
  */

  std::string concatenateStringColumnEntries(
    const ColumnMap& columns,
    const uint64_t row_num,
    const std::vector<std::string> &column_names,
    const std::string delimiter);
  /*
  For each column name, gets the string at the specified row in the column.
  Appends the delimiter to the string. Returns a concatenation of all strings.
  */

  Phrase getStrongPhrase(std::string& s);
  /*
  Returns a single phrase containing all the words from s.
  */

  PhraseCollection getWeakPhrases(std::string& s);
  /*
  Returns a set of natural and chunked phrases from s.
  */

  PhraseCollection sampleFromPhrases(PhraseCollection &phrases,
    uint32_t num_to_sample, uint32_t num_reps);
  /*
  Randomly deletes elements from each phrase, resulting in new phrases.
  Repeats the process num_reps times for each phrase. If a phrase is not
  long enough to choose num_to_sample words, then it is kept but only
  represented once in the output (not num_reps times).
  */

  void mergeStrongWithWeak(PhraseCollection &weak_phrases,
    Phrase &strong_phrase);
  /*
  Concatenates each element from the weak phrases with the strong phrase.
  If _strong_sample_num_words was provided in the constructor, this also
  independently samples from the strong phrase for every weak phrase.
  */

  void validateGreaterThanZero(std::optional<uint32_t> parameter,
    const std::string parameter_name);
  /*
  Throws an error message if the parameter has a value <= 0. The error message
  displays parameter_name.
  */

  // Private constructor for cereal.
  ColdStartTextAugmentation()
      : _strong_column_names(),
        _weak_column_names(),
        _label_column_name(),
        _output_column_name(),
        _weak_min_len(std::nullopt),
        _weak_max_len(std::nullopt),
        _weak_chunk_len(std::nullopt),
        _weak_sample_num_words(std::nullopt),
        _weak_sample_reps(1),
        _strong_max_len(std::nullopt),
        _strong_sample_num_words(std::nullopt),
        _seed(42803){}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Augmentation>(this), _strong_column_names,
            _weak_column_names, _label_column_name, _output_column_name,
            _weak_min_len, _weak_max_len, _weak_chunk_len,
            _weak_sample_num_words, _strong_max_len, _strong_sample_num_words);
  }

};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::ColdStartTextAugmentation)