#pragma once

#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/SparseBatch.h>
#include <schema/InProgressVector.h>
#include <sys/types.h>
#include <fstream>
#include <string_view>
#include "Schema.h"

// Adapted from https://stackoverflow.com/questions/711770/fast-implementation-of-rolling-hash

namespace thirdai::schema {

const uint32_t PRIME_BASE = 257;
const uint32_t PRIME_MOD = 1000000007;

struct CharacterNGramBlock: public ABlock {
  CharacterNGramBlock(const uint32_t col, const uint32_t k, const uint32_t out_dim, const uint32_t offset)
  : _col(col), _k(k), _out_dim(out_dim), _offset(offset) {}

  void extractFeatures(std::vector<std::string_view> line, InProgressSparseVector& vec) override {
    std::vector<uint32_t> indices;

    if (line[_col].size() < _k) {
      return;
    }
    
    int64_t power = 1;
    for (size_t i = 0; i < _k; i++) {
      power = (power * PRIME_BASE) % PRIME_MOD;
    }

    int64_t hash = 0;
    for (size_t i = 0; i < line[_col].size(); i++) {
      // Add last letter
      hash = hash * PRIME_BASE + line[_col][i];
      hash %= PRIME_MOD;

      // Remove first character if needed
      if (i >= _k) {
        hash -= power * line[_col][i - _k] % PRIME_MOD;
        if (hash < 0) {
          hash += PRIME_MOD;
        }
      }

      if (i >= _k - 1) {
        indices.push_back(hash % _out_dim + _offset);
      }
    }

    vec.incrementAtIndices(indices, 1.0);
  }

  static std::shared_ptr<ABlockConfig> Config(const uint32_t col, const uint32_t k, const uint32_t out_dim) {
    return std::make_shared<CharacterNGramBlockConfig>(col, k, out_dim);
  }
  struct CharacterNGramBlockConfig: public ABlockConfig {
    CharacterNGramBlockConfig(const uint32_t col, const uint32_t k, const uint32_t out_dim)
    : _col(col), _k(k), _out_dim(out_dim) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<CharacterNGramBlock>(_col, _k, _out_dim, offset);
      offset += _out_dim;
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t featureDim() const override { return _out_dim; }

  private:
    uint32_t _col;
    uint32_t _k;
    uint32_t _out_dim;
  };

 private:
  
  uint32_t _col;
  uint32_t _k;
  uint32_t _out_dim;
  uint32_t _offset;
};


struct WordNGramBlock: public ABlock {
  WordNGramBlock(const uint32_t col, const uint32_t k, const uint32_t out_dim, const uint32_t offset)
  : _col(col), _k(k), _out_dim(out_dim), _offset(offset) {}

  void extractFeatures(std::vector<std::string_view> line, InProgressSparseVector& vec) override {
    std::vector<uint32_t> indices;

    uint32_t n_words = 1;
    for (const auto& c : line[_col]) { n_words += c == ' ' ? 1 : 0; }
    if (n_words < _k) {
      return;
    }
    std::vector<uint32_t> word_hashes;
    word_hashes.reserve(n_words);

    int64_t word_hash = 0;
    for (const auto& c : line[_col]) {
      // Add last letter
      word_hash = word_hash * PRIME_BASE + c;
      word_hash %= PRIME_MOD;

      if (c == ' ') {
        word_hashes.push_back(word_hash);
        word_hash = 0;
      }
    }
    if (word_hash != 0) {
      word_hashes.push_back(word_hash);
    }

    int64_t power = 1;
    for (size_t i = 0; i < _k; i++) {
      power = (power * PRIME_BASE) % PRIME_MOD;
    }

    int64_t hash = 0;
    for (size_t i = 0; i < word_hashes.size(); i++) {
      // Add last letter
      hash = hash * PRIME_BASE + word_hashes[i];
      hash %= PRIME_MOD;

      // Remove first character if needed
      if (i >= _k) {
        hash -= power * word_hashes[i - _k] % PRIME_MOD;
        if (hash < 0) {
          hash += PRIME_MOD;
        }
      }

      if (i >= _k - 1) {
        indices.push_back(hash % _out_dim + _offset);
      }
    }

    vec.incrementAtIndices(indices, 1.0);
  }

  static std::shared_ptr<ABlockConfig> Config(const uint32_t col, const uint32_t k, const uint32_t out_dim) {
    return std::make_shared<WordNGramBlockConfig>(col, k, out_dim);
  }
  struct WordNGramBlockConfig: public ABlockConfig {
    WordNGramBlockConfig(const uint32_t col, const uint32_t k, const uint32_t out_dim)
    : _col(col), _k(k), _out_dim(out_dim) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      auto built = std::make_unique<WordNGramBlock>(_col, _k, _out_dim, offset);
      offset += _out_dim;
      return built;
    }

    size_t maxColumn() const override { return _col; }

    size_t featureDim() const override { return _out_dim; }

   private:
    uint32_t _col;
    uint32_t _k;
    uint32_t _out_dim;
  };

 private:
  uint32_t _col;
  uint32_t _k;
  uint32_t _out_dim;
  uint32_t _offset;
};


} // namespace thirdai::schema

