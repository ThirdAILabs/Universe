#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace thirdai::dataset {

class Metadata {
 public:
  Metadata(std::vector<BoltVector> vectors, ThreadSafeVocabularyPtr key_vocab,
           uint32_t dim)
      : _vectors(std::move(vectors)),
        _key_vocab(std::move(key_vocab)),
        _dim(dim) {}

  const BoltVector& getVectorForKey(const std::string& key) const {
    return getVectorForUid(_key_vocab->getUid(key));
  }

  const BoltVector& getVectorForUid(uint32_t uid) const {
    return _vectors.at(uid);
  }

  ThreadSafeVocabularyPtr getKeyToUidVocab() const { return _key_vocab; }

  uint32_t featureDim() const { return _dim; }

  static auto make(std::vector<BoltVector> vectors,
                   ThreadSafeVocabularyPtr key_vocab, uint32_t dim) {
    return std::make_shared<Metadata>(std::move(vectors), std::move(key_vocab),
                                      dim);
  }

 private:
  std::vector<BoltVector> _vectors;
  ThreadSafeVocabularyPtr _key_vocab;
  uint32_t _dim;

  // Private constructor for Cereal.
  Metadata() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_vectors, _key_vocab, _dim);
  }
};

using MetadataPtr = std::shared_ptr<Metadata>;

}  // namespace thirdai::dataset