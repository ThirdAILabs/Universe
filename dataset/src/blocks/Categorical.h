#pragma once

#include "BlockInterface.h"
#include "../embeddings/CategoricalEmbeddingModelInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/embeddings/CategoricalEmbeddingModelInterface.h>
#include <dataset/src/embeddings/OneHotEncoding.h>
#include <dataset/src/utils/Conversions.h>
#include <memory>

namespace thirdai::dataset {

/**
 * A block for embedding a sample's raw categorical features.
 */
struct CategoricalBlock : public Block {

  CategoricalBlock(uint32_t col, std::shared_ptr<CategoricalEmbeddingModel>& embedding, bool from_string=false): _col(col), _from_string(from_string), _embedding(embedding) {}

  CategoricalBlock(uint32_t col, uint32_t dim, bool from_string=false): _col(col), _from_string(from_string), _embedding(std::make_shared<OneHotEncoding>(dim)) {}
  
  /**
   * Extracts features from input row and adds it to shared feature vector.
   *
   * Arguments:
   * input_row: a list of columns for a single row.
   * shared_feature_vector: a vector that is shared among all blocks operating on
   *   a particular row. This make it easier for the pipeline object to
   *   concatenate the features produced by each block. 
   * idx_offset: the offset to shift the feature indices by if the preceeding
   *   section of the output vector is occupied by other features.
   */
  void process(const std::vector<std::string>& input_row, BuilderVector& shared_feature_vector, uint32_t idx_offset) final {
    
    const std::string& col_str = input_row[_col];
    uint32_t id = _from_string
      ? hashing::MurmurHash(col_str.c_str(), col_str.length(), 0)
      : getNumberU32(col_str);
    
    _embedding->embedCategory(id, shared_feature_vector, idx_offset);
  };

  /**
   * Returns the dimension of extracted features.
   * This is needed when composing different features into a single vector.
   */
  uint32_t featureDim() final {
    return _embedding->featureDim();
  };

  /**
   * True if the block produces dense features, False otherwise.
   */
  bool isDense() final {
    return _embedding->isDense();
  };

 private:
  uint32_t _col;
  bool _from_string;
  std::shared_ptr<CategoricalEmbeddingModel> _embedding;

};

} // namespace thirdai::dataset