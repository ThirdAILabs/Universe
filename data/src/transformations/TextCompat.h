#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <limits>

namespace thirdai::data {

/**
 * This transformation replicates the behavior of the TextBlock in the old data
 * pipeline, if the block is the only block in the BlockList and the
 * HashedSegmentedFeatureVector is used. The tokenizer and encoder must be the
 * same as the TextBlock. Same for the value of the arg lowercase. The arg
 * encoding_dim must match the dim arg of the text block, and the
 * feature_hash_dim must match the hash_range argument of the BlockList.
 */
class TextCompat final : public Transformation {
 public:
  TextCompat(std::string input_column, std::string output_indices,
             std::string output_values, dataset::TextTokenizerPtr tokenizer,
             dataset::TextEncoderPtr encoder, bool lowercase,
             size_t encoding_dim, size_t hash_range);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  inline uint32_t mimicHashedFeatureVector(uint32_t index) const {
    /**
     * In BlockInterface.h the method addVectorSegment calls
     * addFeatureSegment before addSparseFeatureToSegment is called. This call
     * to addFeatureSegment increments the count of _n_segments_added, thus we
     * combine hashes with 1 instead of 0.
     */
    return hashing::combineHashes(index, 1) % _hash_range;
  }

  std::string _input_column, _output_indices;
  std::string _output_values;

  dataset::TextTokenizerPtr _tokenizer;
  dataset::TextEncoderPtr _encoder;

  bool _lowercase;
  size_t _encoding_dim;
  size_t _hash_range;

  TextCompat() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data