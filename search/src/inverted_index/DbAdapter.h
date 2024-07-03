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

struct DBAdapterConfig {
  std::string db_adapter;
  std::string db_uri;
  uint32_t batch_size = 0;

  DBAdapterConfig()
      : db_adapter("rocksdb") {}

  DBAdapterConfig(std::string uri, uint32_t bulk_update_batch)
      : db_adapter("mongodb"),
        db_uri(std::move(uri)),
        batch_size(bulk_update_batch) {}
};


// TODO(Nicholas): this is specific to the doc count info being serialized into
// a string, this makes sense if the data is returned from a keyvalue store, but
// may not for other DBs. The reason it is done this way instead of converting
// it into a std::vector<DocCount> is to avoid the extra data copies. This could
// be made into an iterable interface to support returning other types of
// iterators for non serialized data.
class SerializedDocCountIterator {
 public:
  explicit SerializedDocCountIterator(std::string&& serialized)
      : _serialized(std::move(serialized)) {
    if (_serialized.size() % sizeof(DocCount) != 0) {
      throw std::runtime_error("Storage is corrupted.");
    }
  }

  size_t len() const { return _serialized.size() / sizeof(DocCount); }

  const auto* begin() const {
    return reinterpret_cast<const DocCount*>(_serialized.data());
  }

  const auto* end() const { return begin() + len(); }

 private:
  std::string _serialized;  // Serialized storage for the blob _data refers to.
};

class DbAdapter {
 public:
  virtual void storeDocLens(const std::vector<DocId>& ids,
                            const std::vector<uint32_t>& doc_lens) = 0;

  virtual void updateTokenToDocs(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_new_docs) = 0;

  virtual std::vector<SerializedDocCountIterator> lookupDocs(
      const std::vector<HashedToken>& query_tokens) = 0;

  virtual uint32_t getDocLen(DocId doc_id) = 0;

  virtual uint64_t getNDocs() = 0;

  virtual uint64_t getSumDocLens() = 0;

  virtual ~DbAdapter() = default;
};

}  // namespace thirdai::search