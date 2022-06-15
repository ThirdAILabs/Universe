#pragma once

#include "Input.h"
#include <bolt/src/graph/Node.h>
#include <numeric>
#include <stdexcept>

namespace thirdai::bolt {

class Concatenate final : public Node {
 public:
  explicit Concatenate()
      : _concatenated_dim(0),
        _sparse_concatenated_dim(0),
        _has_sparse_predecessor(false) {}

  // This may be a no-op, or we may need to map sparse indices to disjoint
  // ranges.
  void forward(uint32_t batch_index, const BoltVector* labels) final {
    (void)labels;

    for (auto& pred : _sparse_predecessors[batch_index]) {
      pred.remapIndices();
    }
  }

  // This may be  no-op or we may need to map disjoint ranges of sparse indices
  // to the dim of each sub-layer.
  void backpropagate(uint32_t batch_index) final { (void)batch_index; }

  BoltVector& getOutput(uint32_t batch_index) final {
    return _outputs[batch_index];
  }

  uint32_t outputDim() const final { return _concatenated_dim; }

  bool hasSparseOutput() const final { return _has_sparse_predecessor; }

  uint32_t sparseOutputDim() const final { return _sparse_concatenated_dim; }

  void addPredecessors(std::vector<NodePtr> inputs) {
    if (!_predecessors.empty()) {
      throw std::invalid_argument("");
    }
    _predecessors = std::move(inputs);
  }

  void initializeState(uint32_t batch_size, bool use_sparsity) final {
    bool sparse = _has_sparse_predecessor && use_sparsity;
    uint32_t dim = sparse ? _sparse_concatenated_dim : _concatenated_dim;
    _outputs = BoltBatch(dim, batch_size, /* is_dense= */ !sparse);

    _sparse_predecessors =
        std::vector<std::vector<SparsePredecesor>>(batch_size);

    // The offset represents where in the concatenated outputs a nodes output
    // is. The neuron_offset is the where in the dimension of the concatenated
    // output a nodes neurons are. These are the same if the nodes are all
    // dense. Otherwise the offset is less than the neuron offset.
    uint32_t output_offset = 0;
    uint32_t neuron_offset = 0;

    for (auto& node : _predecessors) {
      std::vector<BoltVector> node_outputs;
      if (node->hasSparseOutput()) {
        /*
         If the node has a sparse output then we can have it directly
         compute/store the activations and gradients into the concatenated
         BoltVectors, however the active_neurons must be offset such that each
         predecessor has a distinct output range for the output active_neurons
         in the concatenated vectors. Thus we give each sparse predecessor a
         BoltBatch with vectors whose activations and gradients point into the
         vectors of the concatenated batch. The active_neurons are distinct to
         each predecessor so each node gets a unique array to store the
         active_neurons it computes in. Then during the forward pass these
         active_neurons are shifted by the offset of neurons for that node and
         added to the concatenated vector.
        */
        uint32_t node_output_dim = node->sparseOutputDim();

        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
          // Create a datastructure to store the active neurons for the node
          // before the neuron offsets are added.
          _sparse_predecessors[batch_idx].emplace_back(
              node_output_dim,
              _outputs[batch_idx].active_neurons + output_offset,
              neuron_offset);

          // The nodes output will be the active_neurons array just allocated,
          // and pointers into the concatenated activations and gradients.
          node_outputs.push_back(BoltVector(
              _sparse_predecessors[batch_idx].back().activeNeuronsPtr(),
              _outputs[batch_idx].activations + output_offset,
              _outputs[batch_idx].gradients + output_offset, node_output_dim));
        }

      } else {
        // If the node is dense we can just take pointers into the concatenated
        // activations and gradients for its output.
        uint32_t node_output_dim = node->outputDim();
        for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
          node_outputs.push_back(BoltVector(
              nullptr, _outputs[batch_idx].activations + output_offset,
              _outputs[batch_idx].gradients + output_offset, node_output_dim));

          // If the concatenated output is sparse but the node is dense then we
          // need to add indices to its outputs. These will just be 0,1,2,...N
          // and will not change so we can just define them once here.
          if (sparse) {
            uint32_t* start_ptr =
                _outputs[batch_idx].active_neurons + output_offset;

            std::iota(start_ptr, start_ptr + node_output_dim, neuron_offset);
          }
        }
      }
      // Do something with node_outputs

      output_offset +=
          use_sparsity ? node->sparseOutputDim() : node->outputDim();
      neuron_offset += node->outputDim();
    }
  }

  void enqueuePredecessors(std::queue<NodePtr>& nodes) final {
    for (auto& node : _predecessors) {
      nodes.push(node);
    }
  }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    for (auto& node : _predecessors) {
      node->addSparseLayers(sparse_layers);
    }
  }

 protected:
  void compile() final {
    for (const auto& node : _predecessors) {
      bool is_input = dynamic_cast<Input*>(node.get()) != nullptr;
      if (is_input) {
        throw std::invalid_argument(
            "Input layer cannot be part of a Concatenate layer");
      }

      _concatenated_dim += node->outputDim();
      _sparse_concatenated_dim += node->sparseOutputDim();
      _has_sparse_predecessor =
          _has_sparse_predecessor || node->hasSparseOutput();
    }
  }

 private:
  class SparsePredecesor {
    std::vector<uint32_t> _input_active_neurons;
    uint32_t* _output_destination;
    uint32_t _offset;

   public:
    SparsePredecesor(uint32_t dim, uint32_t* dest, uint32_t offset)
        : _input_active_neurons(dim),
          _output_destination(dest),
          _offset(offset) {}

    uint32_t* activeNeuronsPtr() { return _input_active_neurons.data(); }

    void remapIndices() {
      for (uint32_t i = 0; i < _input_active_neurons.size(); i++) {
        _output_destination[i] = _input_active_neurons[i] + _offset;
      }
    }
  };

  std::vector<NodePtr> _predecessors;

  // This is a vector of length batch_size that contains contains information
  // about the sparse nodes for each batch index.
  std::vector<std::vector<SparsePredecesor>> _sparse_predecessors;

  uint32_t _concatenated_dim;
  uint32_t _sparse_concatenated_dim;
  bool _has_sparse_predecessor;

  BoltBatch _outputs;
};

};  // namespace thirdai::bolt