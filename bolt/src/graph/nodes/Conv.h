#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/ConvLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <tuple>

namespace thirdai::bolt {

/**
 * Creates a ConvLayer from inputs. Expects predecessor to have information
 * about the output if interpreted as 3D (currently only Input3D or another
 * ConvNode have this logic).
 */
class ConvNode final : public Node,
                       public std::enable_shared_from_this<ConvNode> {
 private:
  ConvNode(uint64_t num_filters, const std::string& activation,
           std::pair<uint32_t, uint32_t> kernel_size,
           std::pair<uint32_t, uint32_t> next_kernel_size);

  ConvNode(uint64_t num_filters, float sparsity, const std::string& activation,
           std::pair<uint32_t, uint32_t> kernel_size,
           std::pair<uint32_t, uint32_t> next_kernel_size);

  ConvNode(uint64_t num_filters, float sparsity, const std::string& activation,
           std::pair<uint32_t, uint32_t> kernel_size,
           std::pair<uint32_t, uint32_t> next_kernel_size,
           SamplingConfigPtr sampling_config);

 public:
  static std::shared_ptr<ConvNode> makeDense(
      uint32_t num_filters, const std::string& activation,
      std::pair<uint32_t, uint32_t> kernel_size,
      std::pair<uint32_t, uint32_t> next_kernel_size);

  static std::shared_ptr<ConvNode> makeAutotuned(
      uint32_t num_filters, float sparsity, const std::string& activation,
      std::pair<uint32_t, uint32_t> kernel_size,
      std::pair<uint32_t, uint32_t> next_kernel_size);

  static std::shared_ptr<ConvNode> make(
      uint32_t num_filters, float sparsity, const std::string& activation,
      std::pair<uint32_t, uint32_t> kernel_size,
      std::pair<uint32_t, uint32_t> next_kernel_size,
      SamplingConfigPtr sampling_config);

  std::shared_ptr<ConvNode> addPredecessor(NodePtr node);

  uint32_t outputDim() const final;

  bool isInputNode() const final { return false; }

  void initOptimizer() final { _layer->initOptimizer(); }

  bool hasParameters() final { return true; }

 private:
  void compileImpl() final;

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final { return (*_outputs)[0].len; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final;

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_predecessor};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  std::string type() const final { return "conv"; }

  NodeState getState() const final;

  std::tuple<uint32_t, uint32_t, uint32_t> getPredecessorOutputDim() const;

  uint32_t getSparsity() const;

  std::tuple<uint32_t, uint32_t, uint32_t> getOutputDim3D() const;

  // Private constructor for cereal.
  ConvNode() : _config(std::nullopt), _outputs(std::nullopt) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  // One of _layer and _config will always be nullptr/nullopt while the
  // other will contain data
  std::shared_ptr<ConvLayer> _layer;
  std::optional<ConvLayerConfig> _config;
  std::optional<BoltBatch> _outputs;

  std::shared_ptr<Node> _predecessor;
};

using ConvNodePtr = std::shared_ptr<ConvNode>;

}  // namespace thirdai::bolt