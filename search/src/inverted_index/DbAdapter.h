#pragma once

#include <search/src/inverted_index/InvertedIndex.h>
#include <stdexcept>
#include <unordered_set>

namespace thirdai::search {

using DocId = uint64_t;
using HashedToken = uint32_t;

struct DocCount {
  DocCount(DocId doc_id, uint32_t count) : doc_id(doc_id), count(count) {}

  DocId doc_id : 40;
  uint32_t count : 24;
};

#if !_WIN32
// This does not get packed into a single word on windows unlike linux and mac.
static_assert(sizeof(DocCount) == 8);
#endif

class DbAdapter {
 public:
  DbAdapter() {
    if constexpr (sizeof(DocCount) != 8) {
      throw std::invalid_argument(
          "on_disk options are not supported on windows.");
    }
  }
  virtual void storeDocLens(const std::vector<DocId>& ids,
                            const std::vector<uint32_t>& doc_lens) = 0;

  virtual void updateTokenToDocs(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_new_docs) = 0;

  // Used only for update method
  virtual void incrementDocLens(
      const std::vector<DocId>& ids,
      const std::vector<uint32_t>& doc_len_increments) = 0;

  // Used only for update method
  virtual void incrementDocTokenCounts(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_doc_updates) = 0;

  virtual std::vector<std::vector<DocCount>> lookupDocs(
      const std::vector<HashedToken>& query_tokens) const = 0;

  virtual void prune(uint64_t max_docs_with_token) {
    (void)max_docs_with_token;
  }

  virtual void removeDocs(const std::unordered_set<DocId>& docs) = 0;

  virtual uint64_t getDocLen(DocId doc_id) const = 0;

  virtual uint64_t getNDocs() const = 0;

  virtual uint64_t getSumDocLens() const = 0;

  virtual void save(const std::string& save_path) const = 0;

  virtual std::string type() const = 0;

  virtual ~DbAdapter() = default;
};

}  // namespace thirdai::search