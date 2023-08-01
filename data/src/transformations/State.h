#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <dataset/src/mach/MachIndex.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

using dataset::ThreadSafeVocabularyPtr;
using dataset::mach::MachIndexPtr;

class State {
 public:
  explicit State(MachIndexPtr mach_index)
      : _mach_index(std::move(mach_index)) {}

  State() {}

  const auto& machIndex() const {
    if (!_mach_index) {
      throw std::invalid_argument(
          "Transformation state does not contain MachIndex.");
    }
    return _mach_index;
  }

  void setMachIndex(MachIndexPtr new_index) {
    if (_mach_index->numBuckets() != new_index->numBuckets()) {
      throw std::invalid_argument(
          "Output range mismatch in new index. Index output range should be " +
          std::to_string(_mach_index->numBuckets()) +
          " but provided an index with range = " +
          std::to_string(new_index->numBuckets()) + ".");
    }

    _mach_index = std::move(new_index);
  }

  bool containsVocab(const std::string& key) const {
    return _vocabs.count(key);
  }

  void addVocab(const std::string& key, ThreadSafeVocabularyPtr&& vocab) {
    _vocabs.emplace(key, std::move(vocab));
  }

  ThreadSafeVocabularyPtr& getVocab(const std::string& key) {
    if (!_vocabs.count(key)) {
      throw std::invalid_argument("Cannot find vocab for key '" + key + "'.");
    }
    return _vocabs.at(key);
  }

 private:
  MachIndexPtr _mach_index = nullptr;

  std::unordered_map<std::string, ThreadSafeVocabularyPtr> _vocabs;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_mach_index);
  }
};

using StatePtr = std::shared_ptr<State>;

}  // namespace thirdai::data