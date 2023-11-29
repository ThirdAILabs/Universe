#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <limits>

namespace thirdai::data {

class TextCompat final : public Transformation {
 public:
  TextCompat(std::string input_column, std::string output_indices,
             std::string output_values, dataset::TextTokenizerPtr tokenizer,
             dataset::TextEncoderPtr encoder, bool lowercase,
             size_t encoding_dim, size_t feature_hash_dim);

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  inline uint32_t mimicHashedFeatureVector(uint32_t index) const {
    index %= _encoding_dim;
    return hashing::combineHashes(index, 1) % _feature_hash_dim;
  }

  std::string _input_column, _output_indices;
  std::string _output_values;

  dataset::TextTokenizerPtr _tokenizer;
  dataset::TextEncoderPtr _encoder;

  bool _lowercase;
  size_t _encoding_dim;
  size_t _feature_hash_dim;

  TextCompat() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data