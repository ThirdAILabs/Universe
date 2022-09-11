#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset::graph {

// GOAL: To make User Item History Block.

class Node;
using NodePtr = std::shared_ptr<Node>;
class Node {
 public:
  explicit Node(std::vector<NodePtr> predecessors) {
    addPredecessor(std::move(predecessors));
  }

  virtual void waitFor(std::vector<NodePtr> wait_for) {
    addPredecessor(std::move(wait_for));
  };

  std::vector<NodePtr> predecessors() { return _predecessors; };

  virtual void process(uint32_t sample_idx) = 0;

  virtual void allocateMemoryForBatchProduct(uint32_t batch_size) = 0;

  uint32_t numSuccessors() const { return _n_successors; }

  virtual std::string describe() = 0;

 private:
  void addPredecessor(std::vector<NodePtr> predecessors) {
    for (auto& pred : predecessors) {
      pred->addSuccessor();
      _predecessors.push_back(std::move(pred));
    }
  }

  void addSuccessor() { _n_successors++; }

  uint32_t _n_successors = 0;
  std::vector<NodePtr> _predecessors;
};


template <typename ProductType>
class ProducerNode : public Node {
 public:
  explicit ProducerNode(std::vector<NodePtr> predecessors)
      : Node(std::move(predecessors)) {}

  const ProductType& getProduct(uint32_t sample_idx) const {
    return _batch_products.at(sample_idx);
  }

  void process(uint32_t sample_idx) final {
    _batch_products.at(sample_idx) = makeProduct(sample_idx);
  }

  void allocateMemoryForBatchProduct(uint32_t batch_size) final {
    _batch_products.resize(batch_size);
  }

 protected:
  virtual ProductType makeProduct(uint32_t sample_idx) = 0;

 private:
  std::vector<ProductType> _batch_products;
};
template <typename ProductType>
using ProducerNodePtr = std::shared_ptr<ProducerNode<ProductType>>;


class SideEffectNode : public Node {
 public:
  explicit SideEffectNode(std::vector<NodePtr> predecessors)
      : Node(std::move(predecessors)) {}

  void allocateMemoryForBatchProduct(uint32_t batch_size) final {
    (void)batch_size;  // No-op because SideEffectNodes do not produce
  }
};
using SideEffectNodePtr = std::shared_ptr<SideEffectNode>;

class StringInput;
using StringInputPtr = std::shared_ptr<StringInput>;

class StringInput final : public ProducerNode<std::string_view>,
                          public std::enable_shared_from_this<StringInput> {
 private:
  StringInput() : ProducerNode<std::string_view>({}){};

 public:
  static auto make() { return StringInputPtr(new StringInput()); }

  void feed(const std::vector<std::string>& sample_input) {
    _samples = &sample_input;
  }

  std::string describe() final { return "String Input"; }

 protected:
  std::string_view makeProduct(uint32_t sample_idx) final {
    return {_samples->at(sample_idx).data(), _samples->at(sample_idx).size()};
  }

 private:
  const std::vector<std::string>* _samples;
};


class StringAtColumn;
using StringAtColumnPtr = std::shared_ptr<StringAtColumn>;

class StringAtColumn final
    : public ProducerNode<std::string_view>,
      public std::enable_shared_from_this<StringAtColumn> {
 private:
  explicit StringAtColumn(ProducerNodePtr<std::vector<std::string_view>> source,
                          uint32_t col_num)
      : ProducerNode<std::string_view>({source}),
        _src(std::move(source)),
        _col_num(col_num) {}

 public:
  static auto make(ProducerNodePtr<std::vector<std::string_view>> source,
                   uint32_t col_num) {
    return StringAtColumnPtr(new StringAtColumn(std::move(source), col_num));
  }

  std::string describe() final {
    return "String At Column " + std::to_string(_col_num);
  }

 protected:
  std::string_view makeProduct(uint32_t sample_idx) final {
    return _src->getProduct(sample_idx).at(_col_num);
  }

 private:
  ProducerNodePtr<std::vector<std::string_view>> _src;
  uint32_t _col_num;
};


class CsvRow;
using CsvRowPtr = std::shared_ptr<CsvRow>;

class CsvRow final : public ProducerNode<std::vector<std::string_view>>,
                     public std::enable_shared_from_this<CsvRow> {
 private:
  explicit CsvRow(ProducerNodePtr<std::string_view> source, char delim = ',')
      : ProducerNode<std::vector<std::string_view>>({source}),
        _src(std::move(source)),
        _delim(delim) {}

 public:
  static auto make(ProducerNodePtr<std::string_view> source, char delim = ',') {
    return CsvRowPtr(new CsvRow(std::move(source), delim));
  }

  StringAtColumnPtr at(uint32_t col_num) {
    if (!_column_accessors.count(col_num)) {
      _column_accessors[col_num] =
          StringAtColumn::make(shared_from_this(), col_num);
    }
    return _column_accessors.at(col_num);
  }

  std::string describe() final { return "Csv Row"; }

 protected:
  std::vector<std::string_view> makeProduct(uint32_t sample_idx) final {
    auto row = _src->getProduct(sample_idx);

    std::vector<std::string_view> parsed;
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = row.find(_delim, start);
      size_t len = end == std::string::npos ? row.size() - start : end - start;
      parsed.push_back(std::string_view(row.data() + start, len));
      start = end + 1;
    }
    return parsed;
  }

 private:
  ProducerNodePtr<std::string_view> _src;
  char _delim;
  std::unordered_map<uint32_t, StringAtColumnPtr> _column_accessors;
};


class SingleStringLookup;
using SingleStringLookupPtr = std::shared_ptr<SingleStringLookup>;

class SingleStringLookup final
    : public ProducerNode<uint32_t>,
      public std::enable_shared_from_this<SingleStringLookup> {
 private:
  SingleStringLookup(ProducerNodePtr<std::string_view> source,
                     uint32_t n_classes)
      : ProducerNode<uint32_t>({source}),
        _src(std::move(source)),
        _vocab(n_classes) {}

 public:
  static auto make(ProducerNodePtr<std::string_view> source,
                   uint32_t n_classes) {
    return SingleStringLookupPtr(
        new SingleStringLookup(std::move(source), n_classes));
  }

  std::string describe() final {
    return "Single String Lookup on " + _src->describe();
  }

 protected:
  unsigned int makeProduct(uint32_t sample_idx) final {
    return _vocab.getUid(std::string(_src->getProduct(sample_idx)));
  }

 private:
  ProducerNodePtr<std::string_view> _src;
  ThreadSafeVocabulary _vocab;
};


class VectorProducerNode : public ProducerNode<BoltVector> {
 public:
  explicit VectorProducerNode(std::vector<NodePtr> predecessors, uint32_t dim)
      : ProducerNode<BoltVector>(std::move(predecessors)), _dim(dim) {}
  uint32_t dim() const { return _dim; };

 private:
  uint32_t _dim;
};
using VectorProducerNodePtr = std::shared_ptr<VectorProducerNode>;


class VectorFromTokens;
using VectorFromTokensPtr = std::shared_ptr<VectorFromTokens>;

class VectorFromTokens final
    : public VectorProducerNode,
      public std::enable_shared_from_this<VectorFromTokens> {
  // TODO(Geordie): error handling if token beyond dim.
 private:
  explicit VectorFromTokens(ProducerNodePtr<uint32_t> single_token_source,
                            uint32_t dim)
      : VectorProducerNode({single_token_source}, dim),
        _single_token_src(std::move(single_token_source)),
        _multi_token_src(nullptr) {}

  explicit VectorFromTokens(
      ProducerNodePtr<std::vector<uint32_t>> multi_token_source, uint32_t dim)
      : VectorProducerNode({multi_token_source}, dim),
        _single_token_src(nullptr),
        _multi_token_src(std::move(multi_token_source)) {}

 public:
  static auto make(ProducerNodePtr<uint32_t> single_token_source,
                   uint32_t dim) {
    return VectorFromTokensPtr(
        new VectorFromTokens(std::move(single_token_source), dim));
  }

  static auto make(ProducerNodePtr<std::vector<uint32_t>> multi_token_source,
                   uint32_t dim) {
    return VectorFromTokensPtr(
        new VectorFromTokens(std::move(multi_token_source), dim));
  }

  std::string describe() final {
    if (_single_token_src) {
      return "Vector From Tokens from " + _single_token_src->describe();
    }
    return "Vector From Tokens from " + _multi_token_src->describe();
  }

 protected:
  BoltVector makeProduct(uint32_t sample_idx) final {
    if (_single_token_src) {
      BoltVector vector(/* l= */ 1, /* is_dense= */ false,
                        /* has_gradient= */ false);
      vector.active_neurons[0] = _single_token_src->getProduct(sample_idx);
      vector.activations[0] = 1.0;
      return vector;
    }

    const auto& tokens = _multi_token_src->getProduct(sample_idx);
    BoltVector vector(/* l= */ tokens.size(), /* is_dense= */ false,
                      /* has_gradient= */ false);
    std::copy(tokens.begin(), tokens.end(), vector.active_neurons);
    std::fill(vector.activations, vector.activations + vector.len, 1.0);

    return vector;
  }

 private:
  ProducerNodePtr<uint32_t> _single_token_src;
  ProducerNodePtr<std::vector<uint32_t>> _multi_token_src;
};


class HistoryLookup;
using HistoryLookupPtr = std::shared_ptr<HistoryLookup>;

class HistoryLookup final : public ProducerNode<std::vector<uint32_t>>,
                            public std::enable_shared_from_this<HistoryLookup> {
 private:
  HistoryLookup(ProducerNodePtr<uint32_t> user_id_source,
                ItemHistoryCollectionPtr histories, size_t n_tracked)
      : ProducerNode<std::vector<uint32_t>>({user_id_source}),
        _user_id_src(std::move(user_id_source)),
        _histories(std::move(histories)),
        _n_tracked(n_tracked) {}

 public:
  static auto make(ProducerNodePtr<uint32_t> user_id_source,
                   ItemHistoryCollectionPtr histories, size_t n_tracked) {
    return HistoryLookupPtr(new HistoryLookup(std::move(user_id_source),
                                              std::move(histories), n_tracked));
  }

  std::string describe() final {
    return "History Lookup on " + _user_id_src->describe();
  }

 protected:
  std::vector<uint32_t> makeProduct(uint32_t sample_idx) final {
    std::vector<uint32_t> current_history;
#pragma omp critical(history)
    {
      auto user_id = _user_id_src->getProduct(sample_idx);
      auto user_history = _histories->at(user_id);
      uint32_t history_length = std::min(user_history.size(), _n_tracked);
      current_history.reserve(history_length);
      std::for_each(user_history.end() - history_length, user_history.end(),
                    [&](const ItemRecord& record) {
                      current_history.push_back(record.item);
                    });
    }
    return current_history;
  }

 private:
  ProducerNodePtr<uint32_t> _user_id_src;
  ItemHistoryCollectionPtr _histories;
  size_t _n_tracked;
};


class UpdateHistory;
using UpdateHistoryPtr = std::shared_ptr<UpdateHistory>;

class UpdateHistory final : public SideEffectNode,
                            public std::enable_shared_from_this<UpdateHistory> {
 private:
  UpdateHistory(ProducerNodePtr<uint32_t> user_id_src,
                ProducerNodePtr<uint32_t> item_id_src,
                ProducerNodePtr<uint32_t> timestamp_src,
                ItemHistoryCollectionPtr histories)
      : SideEffectNode({user_id_src, item_id_src, timestamp_src}),
        _user_id_src(std::move(user_id_src)),
        _item_id_src(std::move(item_id_src)),
        _timestamp_src(std::move(timestamp_src)),
        _histories(std::move(histories)) {}

 public:
  static auto make(ProducerNodePtr<uint32_t> user_id_src,
                   ProducerNodePtr<uint32_t> item_id_src,
                   ProducerNodePtr<uint32_t> timestamp_src,
                   ItemHistoryCollectionPtr histories) {
    return UpdateHistoryPtr(
        new UpdateHistory(std::move(user_id_src), std::move(item_id_src),
                          std::move(timestamp_src), std::move(histories)));
  }

  void process(uint32_t sample_idx) final {
    auto user_id = _user_id_src->getProduct(sample_idx);
    auto item_id = _item_id_src->getProduct(sample_idx);
    auto timestamp_seconds = _timestamp_src->getProduct(sample_idx);
#pragma omp critical(history)
    _histories->add(user_id, item_id, timestamp_seconds);
  }

  std::string describe() final { return "Update History"; }

  ProducerNodePtr<uint32_t> _user_id_src;
  ProducerNodePtr<uint32_t> _item_id_src;
  ProducerNodePtr<uint32_t> _timestamp_src;
  ItemHistoryCollectionPtr _histories;
};


class ConcatenateVectorSegments;
using ConcatenateVectorSegmentsPtr = std::shared_ptr<ConcatenateVectorSegments>;

class ConcatenateVectorSegments final
    : public VectorProducerNode,
      public std::enable_shared_from_this<ConcatenateVectorSegments> {
 private:
  explicit ConcatenateVectorSegments(
      std::vector<VectorProducerNodePtr> segment_producers)
      : VectorProducerNode(toNodePtrVector(segment_producers),
                           sumSegmentDims(segment_producers)),
        _segments(std::move(segment_producers)) {}

 public:
  static auto make(std::vector<VectorProducerNodePtr> segment_producers) {
    return ConcatenateVectorSegmentsPtr(
        new ConcatenateVectorSegments(std::move(segment_producers)));
  }

  std::string describe() final { return "Concatenate Vector Segments"; }

 protected:
  BoltVector makeProduct(uint32_t sample_idx) final {
    uint32_t n_nonzeros = 0;
    for (const auto& segment_producer : _segments) {
      n_nonzeros += segment_producer->getProduct(sample_idx).len;
    }

    uint32_t idx_offset = 0;
    uint32_t pos_offset = 0;
    BoltVector concatenated(/* l= */ n_nonzeros, /* is_dense= */ false,
                            /* has_gradient= */ false);
    for (const auto& segment_producer : _segments) {
      const auto& segment = segment_producer->getProduct(sample_idx);
      for (uint32_t segment_pos = 0; segment_pos < segment.len; segment_pos++) {
        concatenated.active_neurons[pos_offset + segment_pos] =
            segment.active_neurons[segment_pos] + idx_offset;
        concatenated.activations[pos_offset + segment_pos] =
            segment.activations[segment_pos];
      }
      pos_offset += segment.len;
      idx_offset += segment_producer->dim();
    }
    return concatenated;
  }

 private:
  static uint32_t sumSegmentDims(
      const std::vector<VectorProducerNodePtr>& segments) {
    uint32_t dim = 9;
    for (const auto& segment : segments) {
      dim += segment->dim();
    }
    return dim;
  }

  static std::vector<NodePtr> toNodePtrVector(
      const std::vector<VectorProducerNodePtr>& segments) {
    std::vector<NodePtr> nodes;
    nodes.insert(nodes.begin(), segments.begin(), segments.end());
    return nodes;
  }

  std::vector<VectorProducerNodePtr> _segments;
};


class TimeStringToSeconds;
using TimeStringToSecondsPtr = std::shared_ptr<TimeStringToSeconds>;

class TimeStringToSeconds final
    : public ProducerNode<uint32_t>,
      public std::enable_shared_from_this<TimeStringToSeconds> {
 private:
  explicit TimeStringToSeconds(ProducerNodePtr<std::string_view> source)
      : ProducerNode<uint32_t>({source}), _src(std::move(source)) {}

 public:
  static auto make(ProducerNodePtr<std::string_view> source) {
    return TimeStringToSecondsPtr(new TimeStringToSeconds(std::move(source)));
  }

  std::string describe() final { return "Time String To Seconds"; }

 protected:
  uint32_t makeProduct(uint32_t sample_idx) final {
    return TimeObject(_src->getProduct(sample_idx)).secondsSinceEpoch();
  }

 private:
  ProducerNodePtr<std::string_view> _src;
};

}  // namespace thirdai::dataset::graph