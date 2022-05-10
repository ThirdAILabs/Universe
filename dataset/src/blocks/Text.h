#pragma once

#include "../encodings/text/BoltTokenizer.h"
#include "../encodings/text/TextEncodingInterface.h"
#include "BlockInterface.h"
#include <memory>

namespace thirdai::dataset {

/**
 * A block for embedding a sample's raw textual features.
 */
struct TextBlock : public Block {
  /**
   * Constructor.
   *
   * Arguments:
   *   column: int - given a sample as a row of raw features, column is the
   * position of the text to be embedded within this row; "we want to embed the
   * text in column n of the sample." embedding_model: TextEmbeddingModel - the
   * model for encoding the text as vectors. pipeline: list of functions that
   * accept and return a list of strings - a pipeline for preprocessing string
   * tokens before they get encoded by the embedding model.
   */
  TextBlock(uint32_t col, std::shared_ptr<TextEncoding>& encoding)
      : _col(col), _encoding(encoding) {}

  TextBlock(uint32_t col, uint32_t dim)
      : _col(col), _encoding(std::make_shared<BoltTokenizer>(dim)) {}

  /**
   * Extracts features from input row and adds it to shared feature vector.
   *
   * Arguments:
   * input_row: a list of columns for a single row.
   * shared_feature_vector: a vector that is shared among all blocks operating
   * on a particular row. This make it easier for the pipeline object to
   *   concatenate the features produced by each block.
   * idx_offset: the offset to shift the feature indices by if the preceeding
   *   section of the output vector is occupied by other features.
   */
  void process(const std::vector<std::string>& input_row,
               BuilderVector& shared_feature_vector,
               uint32_t idx_offset) final {
    _encoding->embedText(input_row[_col], shared_feature_vector, idx_offset);
  };

  /**
   * Returns the dimension of extracted features.
   * This is needed when composing different features into a single vector.
   */
  uint32_t featureDim() final { return _encoding->featureDim(); };

  /**
   * True if the block produces dense features, False otherwise.
   */
  bool isDense() final { return _encoding->isDense(); };

 private:
  uint32_t _col;
  std::shared_ptr<TextEncoding> _encoding;
};

}  // namespace thirdai::dataset