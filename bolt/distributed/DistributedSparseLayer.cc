#include "DistributedSparseLayer.h"
#include <cmath>
#include <mpi.h>
#include <vector>

namespace thirdai::bolt {

DistributedSparseLayer::DistributedSparseLayer(uint64_t dim, uint64_t prev_dim,
                                               float sparsity,
                                               ActivationFunc act_func,
                                               SamplingConfig sampling_config)
    : _batch_size(0), _full_dim(dim), _act_func(act_func) {
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_world_size);

  _local_dim = _full_dim / _world_size;
  uint32_t rem = _full_dim % _world_size;
  _neuron_offset =
      _local_dim * _rank + std::min<uint32_t>(_rank, _full_dim % rem);
  if (_rank < rem) {
    _local_dim++;
  }

  ActivationFunc act_func_out = _act_func == ActivationFunc::Softmax
                                    ? ActivationFunc::DistributedSoftmax
                                    : _act_func;

  _local_layer = new SparseLayer(_local_dim, prev_dim, sparsity, act_func_out,
                                 sampling_config);
}

void DistributedSparseLayer::ReduceErrors() {
  for (uint32_t b = 0; b < _batch_size; b++) {
    MPI_Allreduce(MPI_IN_PLACE, _errors[b], _total_active_lens[b], MPI_FLOAT,
                  MPI_SUM, MPI_COMM_WORLD);

    float* start = _errors[b] + _active_offsets[b][_rank];
    std::copy(start, start + _active_lens[b][_rank],
              _local_layer->GetErrors(b));
  }
}

void DistributedSparseLayer::GatherActivations() {
  for (uint32_t b = 0; b < _batch_size; b++) {
    _active_lens[b][_rank] = _local_layer->GetLen(b);
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, _active_lens[b], 1,
                  MPI_UNSIGNED, MPI_COMM_WORLD);

    _active_offsets[b][0] = 0;
    _total_active_lens[b] = _active_lens[b][0];
    for (uint32_t i = 1; i < _world_size; i++) {
      uint32_t len = _active_lens[b][i];
      _active_offsets[b][i] =
          _active_offsets[b][i - 1] + _active_lens[b][i - 1];
    }

    uint32_t len = _local_layer->GetLen(b);
    const uint32_t* local_active = _local_layer->GetIndices(b);
    uint32_t* global_active = _active_neurons[b] + _active_offsets[b][_rank];
    for (uint32_t i = 0; i < len; i++) {
      global_active[i] = local_active[i];
    }
    std::copy(_local_layer->GetValues(b),
              _local_layer->GetValues(b) + _local_layer->GetLen(b),
              _activations[b] + _active_offsets[b][_rank]);

    MPI_Allgatherv(MPI_IN_PLACE, len, MPI_UNSIGNED, _active_neurons[b],
                   _active_lens[b], _active_offsets[b], MPI_UNSIGNED,
                   MPI_COMM_WORLD);

    float* local_activations = (float*)_local_layer->GetValues(b);
    if (_act_func == ActivationFunc::Softmax) {
      // 1. Get local max activation
      float max_act = 0;
      for (uint32_t i = 0; i < len; i++) {
        if (local_activations[i] > max_act) {
          max_act = local_activations[i];
        }
      }
      // 2. Allreduce to get global max activation
      float overall_max_act;
      MPI_Allreduce(&max_act, &overall_max_act, 1, MPI_FLOAT, MPI_MAX,
                    MPI_COMM_WORLD);
      // 3. Compute local total activation and raise local values to exponent
      float total = 0;
      for (uint32_t i = 0; i < len; i++) {
        float new_act = exp(local_activations[i] - overall_max_act);
        local_activations[i] = new_act;
        total += new_act;
      }
      // 4. Reduce overall total
      float overall_total = 0;
      MPI_Allreduce(&total, &overall_total, 1, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);

      // 5. Divide by total to get final activations.
      for (uint32_t i = 0; i < len; i++) {
        local_activations[i] /= (overall_total + EPS);
      }
    }

    MPI_Allgatherv(_local_layer->GetValues(b), len, MPI_FLOAT, _activations[b],
                   _active_lens[b], _active_offsets[b], MPI_FLOAT,
                   MPI_COMM_WORLD);

    std::fill_n(_errors[b], _full_dim, 0);
  }
}

void DistributedSparseLayer::ComputeErrors(uint32_t batch_indx,
                                           const uint32_t* labels,
                                           uint32_t label_len) {
  std::vector<uint32_t> local_labels;
  for (uint32_t i = 0; i < label_len; i++) {
    if (_neuron_offset <= labels[i] &&
        labels[i] < _neuron_offset + _local_dim) {
      local_labels.push_back(labels[i] - _neuron_offset);
    }
  }
  _local_layer->ComputeErrors(batch_indx, local_labels.data(),
                              local_labels.size());
}

void DistributedSparseLayer::SetBatchSize(uint64_t new_batch_size) {
  _local_layer->SetBatchSize(new_batch_size);

  if (new_batch_size == _batch_size) {
    return;
  }

  for (uint64_t batch = 0; batch < _batch_size; batch++) {
    delete[] _active_neurons[batch];
    delete[] _activations[batch];
    delete[] _errors[batch];
  }

  delete[] _active_lens;
  delete[] _active_neurons;
  delete[] _activations;
  delete[] _errors;

  _batch_size = new_batch_size;

  _total_active_lens = new uint32_t[_batch_size];
  _active_lens = new int*[_batch_size];
  _active_offsets = new int*[_batch_size];
  _active_neurons = new uint32_t*[_batch_size];
  _activations = new float*[_batch_size];
  _errors = new float*[_batch_size];

  for (uint64_t batch = 0; batch < _batch_size; batch++) {
    _active_lens[batch] = new int[_world_size];
    _active_offsets[batch] = new int[_world_size];
    _active_neurons[batch] = new uint32_t[_full_dim];
    _activations[batch] = new float[_full_dim];
    _errors[batch] = new float[_full_dim]();
  }
}

}  // namespace thirdai::bolt
