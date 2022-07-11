#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/ContiguousNumericId.h>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace thirdai::dataset {

/**
 * A block that encodes categorical features (e.g. a numerical ID or an
 * identification string).
 */
class CategoricalBlock : public Block {
 public:
  /**
   * Constructor.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   encoding: CategoricalEncoding - the categorical feature encoding model.
   */
  CategoricalBlock(uint32_t col, std::shared_ptr<CategoricalEncoding> encoding,
                   GraphPtr graph = nullptr, size_t max_n_neighbors = 0)
      : _col(col),
        _encoding(std::move(encoding)),
        _graph(std::move(graph)),
        _max_n_neighbors(max_n_neighbors) {
    if (_graph != nullptr && _max_n_neighbors == 0) {
      throw std::invalid_argument(
          "Provided a graph but `max_n_neighbors` is set to 0. This means "
          "graph information will not be used at all.");
    }
  }

  /**
   * Constructor with default encoder.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   dim: int - the dimension of the encoding.
   */
  CategoricalBlock(uint32_t col, uint32_t dim, GraphPtr graph = nullptr,
                   size_t max_n_neighbors = 0)
      : CategoricalBlock(col, std::make_shared<ContiguousNumericId>(dim),
                         std::move(graph), max_n_neighbors) {}

  uint32_t featureDim() const final {
    uint32_t multiplier = _max_n_neighbors > 0 ? 2 : 1;
    return _encoding->featureDim() * multiplier;
  };

  bool isDense() const final { return _encoding->isDense(); };

  uint32_t expectedNumColumns() const final { return _col + 1; };

  std::pair<std::string, uint32_t> giveMessage() const final {
    return std::make_pair(std::string("From the CategoricalBlock"), _col);
  }

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    _encoding->encodeCategory(input_row.at(_col), vec, /* offset = */ 0);
    std::string id(input_row[_col]);
    if (_graph != nullptr && _graph->count(id) > 0) {
      auto& neighbors = _graph->at(id);
      for (size_t i = 0; i < std::min(_max_n_neighbors, neighbors.size());
           i++) {
        std::string_view neighbor_view(neighbors[i].data(),
                                       neighbors[i].size());
        _encoding->encodeCategory(neighbor_view, vec,
                                  /* offset = */ _encoding->featureDim());
      }
    }
  }

 private:
  uint32_t _col;
  std::shared_ptr<CategoricalEncoding> _encoding;
  GraphPtr _graph;
  size_t _max_n_neighbors;
};

}  // namespace thirdai::dataset