#pragma once

#include <search/src/neural_db/Constraints.h>

namespace thirdai::search::ndb {

using ChunkId = uint64_t;
using DocId = std::string;

struct Chunk {
  ChunkId id;
  std::string text;

  std::string document;
  DocId doc_id;
  uint32_t doc_version;

  MetadataMap metadata;

  Chunk(ChunkId id, std::string text, std::string document, DocId doc_id,
        uint32_t doc_version, MetadataMap metadata)
      : id(id),
        text(std::move(text)),
        document(std::move(document)),
        doc_id(std::move(doc_id)),
        doc_version(doc_version),
        metadata(std::move(metadata)) {}
};

struct ChunkData {
  ChunkData() {}

  ChunkData(std::string text, std::string document, DocId doc_id,
            uint32_t doc_version)
      : text(std::move(text)),
        document(std::move(document)),
        doc_id(std::move(doc_id)),
        doc_version(doc_version) {}

  std::string text;

  std::string document;
  DocId doc_id;
  uint32_t doc_version;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(text, document, doc_id, doc_version);
  }
};

struct ChunkCount {
  ChunkCount(ChunkId chunk_id, uint32_t count)
      : chunk_id(chunk_id), count(count) {}

  /**
   * We use 40 bits for the id and 24 bits for the count so that the entire
   * struct fits within a word. This is prefered to uint32_t for each because it
   * allows for more chunk ids, and still supports counts up to 16 million.
   */
  ChunkId chunk_id : 40;
  uint32_t count : 24;
};

static_assert(sizeof(ChunkCount) == 8, "ChunkCount should be 8 bytes.");

}  // namespace thirdai::search::ndb