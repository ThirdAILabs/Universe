#pragma once

#include "../src/Layer.h"
#include "../src/SparseLayer.h"

namespace thirdai::bolt {

class DistributedSparseLayer final : public Layer {
 public:
  DistributedSparseLayer(uint64_t dim, uint64_t prev_dim, float sparsity,
                         ActivationFunc act_func,
                         SamplingConfig sampling_config);

  void FeedForward(uint32_t batch_indx, const uint32_t* indices,
                   const float* values, uint32_t len, uint32_t* labels,
                   uint32_t label_len) override {
    _local_layer->FeedForward(batch_indx, indices, values, len, labels,
                             label_len);
    // Must call GatherActivations after FeedForward
  }

  void GatherActivations();

  void Backpropagate(uint32_t batch_indx, const uint32_t* indices,
                     const float* values, float* errors,
                     uint32_t len) override {
    // Must call ReduceErrors before Backpropagate
    _local_layer->Backpropagate(batch_indx, indices, values, errors, len);
  }

  void BackpropagateFirstLayer(uint32_t batch_indx, const uint32_t* indices,
                               const float* values, float* errors,
                               uint32_t len) override {
    // Must call ReduceErrors before BackpropagateFirstLayer
    _local_layer->BackpropagateFirstLayer(batch_indx, indices, values, errors,
                                         len);
  }

  void ReduceErrors();

  void ComputeErrors(uint32_t batch_indx, const uint32_t* labels,
                     uint32_t label_len) override;

  void UpdateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) override {
    _local_layer->UpdateParameters(lr, iter, B1, B2, eps);
  }

  uint32_t GetLen(uint32_t batch_indx) const override {
    return _total_active_lens[batch_indx];
  }

  const uint32_t* GetIndices(uint32_t batch_indx) const override {
    return _active_neurons[batch_indx];
  }

  const float* GetValues(uint32_t batch_indx) const override {
    return _activations[batch_indx];
  }

  float* GetErrors(uint32_t batch_indx) override { return _errors[batch_indx]; }

  void BuildHashTables() override { _local_layer->BuildHashTables(); }

  void ReBuildHashFunction() override { _local_layer->ReBuildHashFunction(); }

  void SetSparsity(float new_sparsity) override {
    _local_layer->SetSparsity(new_sparsity);
  }

  void SetBatchSize(uint64_t new_batch_size) override;

  void ShuffleRandNeurons() override { _local_layer->ShuffleRandNeurons(); }

 private:
  int _rank, _world_size;

  uint64_t _batch_size, _full_dim, _local_dim, _neuron_offset;
  ActivationFunc _act_func;

  uint32_t* _total_active_lens;
  int** _active_lens;
  int** _active_offsets;
  uint32_t** _active_neurons;
  float** _activations;
  float** _errors;

  SparseLayer* _local_layer;
};

}  // namespace thirdai::bolt