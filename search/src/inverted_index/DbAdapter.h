#pragma once

#include <search/src/inverted_index/InvertedIndex.h>
#include <stdexcept>

namespace thirdai::search {

using DocId = uint64_t;
using HashedToken = uint32_t;

struct __attribute__((packed)) DocCount {
  DocCount(DocId doc_id, uint32_t count) : doc_id(doc_id), count(count) {}

  DocId doc_id;
  uint32_t count;
};

class DocCountIterator {
 public:
  explicit DocCountIterator(std::string&& serialized)
      : _storage(std::move(serialized)) {
    _data = reinterpret_cast<const DocCount*>(_storage.data());

    if (_storage.size() % sizeof(DocCount) != 0) {
      throw std::runtime_error("Storage is corrupted.");
    }

    _len = _storage.size() / sizeof(DocCount);
  }

  size_t len() const { return _len; }

  const auto* begin() const { return _data; }

  const auto* end() const { return _data + _len; }

 private:
  const DocCount* _data;
  size_t _len;

  std::string _storage;  // Serialized storage for the blob _data refers to.
};

class DbAdapter {
 public:
  virtual void storeDocLens(const std::vector<DocId>& ids,
                            const std::vector<uint32_t>& doc_lens) = 0;

  virtual void updateTokenToDocs(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_new_docs) = 0;

  virtual std::vector<DocCountIterator> lookupDocs(
      const std::vector<HashedToken>& query_tokens) const = 0;

  virtual uint32_t getDocLen(DocId doc_id) const = 0;

  virtual uint64_t getNDocs() const = 0;

  virtual uint64_t getSumDocLens() const = 0;

  virtual ~DbAdapter() = default;
};

}  // namespace thirdai::search