#include "BlockInterface.h"
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <utility>
#include <auto_ml/src/dataset_factories/udt/GraphInfo.h>
#include <exceptions/src/Exceptions.h>

namespace thirdai::dataset {

/**
 * Averages the numerical features of 
 *
 */
class NormalizedNeighborVectorsBlock final : public Block {
 public:
  explicit NormalizedNeighborVectorsBlock(ColumnIdentifier node_id_col, automl::data::GraphInfoPtr graph_ptr) : _node_id_col(std::move(node_id_col)), _graph_ptr(std::move(graph_ptr)) {}

  uint32_t featureDim() const final {
    return _graph_ptr->featureDim();
  };

  bool isDense() const final { return true; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final;

  static auto make(ColumnIdentifier col, automl::data::GraphInfoPtr graph_ptr) {
    return std::make_shared<NormalizedNeighborVectorsBlock>(std::move(col), graph_ptr);
  }

 protected:
  std::exception_ptr buildSegment(ColumnarInputSample& input,
                                  SegmentedFeatureVector& vec) final;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_node_id_col};
  }

 private:
  ColumnIdentifier _node_id_col;
  automl::data::GraphInfoPtr _graph_ptr;
};

}  // namespace thirdai::dataset