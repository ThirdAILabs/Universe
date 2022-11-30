#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt/src/graph/Node.h>
#include <bolt_vector/src/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

// A node subclass for input layers. The input batch will be stored in this
// layer so that subsequent layers can access the inputs through its getOutput()
// method. This makes the interface simplier by generalizing the forward pass so
// that other layers always just access the outputs of the previous layer rather
// than have to worry if they they need to access an input directly or access
// the outputs of a previous layer.
class Input final : public Node {
 private:
  explicit Input(uint32_t expected_input_dim,
                 std::optional<std::pair<uint32_t, uint32_t>>
                     num_nonzeros_range = std::nullopt);

 public:
  static std::shared_ptr<Input> make(uint32_t expected_dim) {
    return std::shared_ptr<Input>(new Input(expected_dim));
  }

  static std::shared_ptr<Input> makeTokenInput(
      uint32_t expected_dim, std::pair<uint32_t, uint32_t> num_tokens_range) {
    return std::shared_ptr<Input>(new Input(expected_dim, num_tokens_range));
  }

  // This class does not own this memory, but we pass it in as a pointer that
  // will be stored as a field so it can be used in future method calls. It is
  // only valid until the next time cleanupAfterBatchProcessing is called.
  void setInputs(BoltBatch* inputs);

  uint32_t expectedInputDim() const { return _expected_input_dim; }

  std::optional<std::pair<uint32_t, uint32_t>> numNonZerosRange() const {
    return _num_nonzeros_range;
  }

  uint32_t outputDim() const final { return _expected_input_dim; }

  bool isInputNode() const final { return true; }

  void initOptimizer() final {
    throw std::logic_error("Should not call initOptimizer() on Input node");
  }

 private:
  void compileImpl() final;

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final {
    throw std::logic_error(
        "Cannot know ahead of time the number of nonzeros "
        "in the output of an Input layer.");
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    (void)labels;
    (void)vec_index;
  }

  void backpropagateImpl(uint32_t vec_index) final { (void)vec_index; }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_input_batch)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _input_batch = nullptr; }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  std::string type() const final { return "input"; }

  std::vector<NodePtr> getPredecessorsImpl() const final { return {}; }

  NodeState getState() const final;

  void checkDimForInput(const BoltVector& vec) const;

  // Private constructor for cereal.
  Input() : _num_nonzeros_range(std::nullopt) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  bool _compiled;
  BoltBatch* _input_batch;
  uint32_t _expected_input_dim;
  std::optional<std::pair<uint32_t, uint32_t>> _num_nonzeros_range;
};

using InputPtr = std::shared_ptr<Input>;

}  // namespace thirdai::bolt
