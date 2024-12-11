#pragma once

#include <search/src/neural_db/Constraints.h>

namespace thirdai::search::ndb {

using ChunkId = uint64_t;
using DocId = std::string;

struct NewChunk {
  std::string text;
  MetadataMap metadata;
};

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

  ChunkId chunk_id : 40;
  uint32_t count : 24;
};

}  // namespace thirdai::search::ndb