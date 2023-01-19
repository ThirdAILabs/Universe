#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <exceptions/src/Exceptions.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt {

class SwitchNode;

class FullyConnectedNode final
    : public Node,
      public std::enable_shared_from_this<FullyConnectedNode> {
  friend class SwitchNode;

 private:
  FullyConnectedNode(uint64_t dim, const std::string& activation);

  FullyConnectedNode(uint64_t dim, float sparsity,
                     const std::string& activation);

  FullyConnectedNode(uint64_t dim, float sparsity,
                     const std::string& activation,
                     SamplingConfigPtr sampling_config);

 public:
  static std::shared_ptr<FullyConnectedNode> makeDense(
      uint32_t dim, const std::string& activation);

  static std::shared_ptr<FullyConnectedNode> makeAutotuned(
      uint32_t dim, float sparsity, const std::string& activation);

  static std::shared_ptr<FullyConnectedNode> make(
      uint32_t dim, float sparsity, const std::string& activation,
      SamplingConfigPtr sampling_config);

  static std::shared_ptr<FullyConnectedNode> makeExplicitSamplingConfig(
      uint32_t dim, float sparsity, const std::string& activation,
      uint32_t num_tables, uint32_t hashes_per_table, uint32_t reservoir_size);

  std::shared_ptr<FullyConnectedNode> addPredecessor(NodePtr node);

  uint32_t outputDim() const final;

  bool isInputNode() const final;

  void initOptimizer() final;

  ActivationFunction getActivationFunction() const;

  void saveParameters(const std::string& filename) const;

  void loadParameters(const std::string& filename);

  float getSparsity();

  std::shared_ptr<FullyConnectedNode> setSparsity(float sparsity);

  float* getWeightsPtr();

  float* getBiasesPtr();

  float* getWeightGradientsPtr();

  float* getBiasGradientsPtr();
  void disableSparseParameterUpdates() final;

  void nodeSaveType(bool whether_hard_save) final;

  bool hasParameters() final { return true; }

  std::string type() const final { return "fc"; }

 private:
  void compileImpl() final;

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final;

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final;

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final;

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final;

  std::vector<NodePtr> getPredecessorsImpl() const final;

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  NodeState getState() const final;

  // Private constructor for cereal. Must create dummy config since no default
  // constructor exists for layer config.
  FullyConnectedNode() : _config(std::nullopt) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
  // One of _layer and _config will always be nullptr/nullopt while the
  // other will contain data
  std::shared_ptr<FullyConnectedLayer> _layer;
  std::optional<FullyConnectedLayerConfig> _config;
  std::optional<BoltBatch> _outputs;

  NodePtr _predecessor;
};

using FullyConnectedNodePtr = std::shared_ptr<FullyConnectedNode>;

}  // namespace thirdai::bolt
