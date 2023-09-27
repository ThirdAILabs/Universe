#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <data/src/transformations/State.h>
#include <dataset/src/mach/MachIndex.h>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace thirdai::automl::mach {

static std::vector<std::vector<std::pair<uint32_t, double>>>
rankedEntitiesFromOutputs(const bolt::Tensor& outputs,
                          const dataset::mach::MachIndex& index, uint32_t top_k,
                          uint32_t num_scanned_buckets) {
  uint32_t num_retrieved = std::min(top_k, index.numEntities());
  uint32_t batch_size = outputs.batchSize();

  std::vector<std::vector<std::pair<uint32_t, double>>> predicted_entities(
      batch_size);
#pragma omp parallel for default(none)                                    \
    shared(outputs, predicted_entities, num_retrieved, batch_size, index, \
           num_scanned_buckets) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs.getVector(i);
    auto predictions = index.decode(
        /* output = */ vector,
        /* top_k = */ num_retrieved,
        /* num_buckets_to_eval = */ num_scanned_buckets);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
}

static std::vector<std::vector<uint32_t>> rankedBucketsFromOutputs(
    const bolt::Tensor& outputs, const dataset::mach::MachIndex& index,
    bool force_non_empty, std::optional<uint32_t> num_hashes) {
  uint32_t k = num_hashes.value_or(index.numHashes());

  std::vector<std::vector<uint32_t>> all_hashes(outputs.batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, k, force_non_empty, index)
  for (uint32_t i = 0; i < outputs.batchSize(); i++) {
    const BoltVector& output = outputs.getVector(i);

    TopKActivationsQueue heap;
    if (force_non_empty) {
      heap = index.topKNonEmptyBuckets(output, k);
    } else {
      heap = output.findKLargestActivations(k);
    }

    std::vector<uint32_t> hashes;
    while (!heap.empty()) {
      auto [_, active_neuron] = heap.top();
      hashes.push_back(active_neuron);
      heap.pop();
    }

    std::reverse(hashes.begin(), hashes.end());

    all_hashes[i] = hashes;
  }

  return all_hashes;
}

static std::vector<float> averageBucketEmbeddings(
    uint32_t label, const dataset::mach::MachIndex& index,
    const bolt::FullyConnectedLayer& output_layer) {
  const std::vector<uint32_t>& hashed_neurons = index.getHashes(label);
  std::vector<float> averaged_embedding(output_layer.getInputDim());
  for (uint32_t neuron_id : hashed_neurons) {
    auto weights = output_layer.getWeightsByNeuron(neuron_id);
    if (weights.size() != averaged_embedding.size()) {
      throw std::invalid_argument("Output dim mismatch.");
    }
    for (uint32_t i = 0; i < weights.size(); i++) {
      averaged_embedding[i] += weights[i];
    }
  }

  // TODO(david) try averaging and summing
  for (float& weight : averaged_embedding) {
    weight /= averaged_embedding.size();
  }

  return averaged_embedding;
}

struct BucketScore {
  uint32_t frequency = 0;
  float score = 0.0;
};

struct CompareBuckets {
  bool operator()(const std::pair<uint32_t, BucketScore>& lhs,
                  const std::pair<uint32_t, BucketScore>& rhs) {
    if (lhs.second.frequency == rhs.second.frequency) {
      return lhs.second.score > rhs.second.score;
    }
    return lhs.second.frequency > rhs.second.frequency;
  }
};

// TODO(MACHREFACTOR): why is this soooo long T-T What's going on here?
static std::vector<uint32_t> topHashesForDoc(
    std::vector<TopKActivationsQueue>&& top_k_per_sample,
    dataset::mach::MachIndex& mach_index, uint32_t num_buckets_to_sample,
    uint32_t num_random_hashes) {
  uint32_t num_hashes = mach_index.numHashes();

  if (num_buckets_to_sample < mach_index.numHashes()) {
    throw std::invalid_argument(
        "Sampling from fewer buckets than num_hashes is not supported. If "
        "you'd like to introduce using fewer hashes, please reset the number "
        "of hashes for the index.");
  }

  if (num_buckets_to_sample > mach_index.numBuckets()) {
    throw std::invalid_argument(
        "Cannot sample more buckets than there are in the index.");
  }

  std::unordered_map<uint32_t, BucketScore> hash_freq_and_scores;
  for (auto& top_k : top_k_per_sample) {
    while (!top_k.empty()) {
      auto [activation, active_neuron] = top_k.top();
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, activation};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += activation;
      }
      top_k.pop();
    }
  }

  // We sort the hashes first by number of occurances and tiebreak with
  // the higher aggregated score if necessary. We don't only use the
  // activations since those typically aren't as useful as the
  // frequencies.
  std::vector<std::pair<uint32_t, BucketScore>> sorted_hashes(
      hash_freq_and_scores.begin(), hash_freq_and_scores.end());

  CompareBuckets cmp;
  std::sort(sorted_hashes.begin(), sorted_hashes.end(), cmp);

  if (num_buckets_to_sample > num_hashes) {
    // If we are sampling more buckets then we end up using we rerank the
    // buckets based on size to load balance the index.
    std::sort(sorted_hashes.begin(),
              sorted_hashes.begin() + num_buckets_to_sample,
              [&mach_index, &cmp](const auto& lhs, const auto& rhs) {
                size_t lhs_size = mach_index.bucketSize(lhs.first);
                size_t rhs_size = mach_index.bucketSize(rhs.first);

                // Give preference to emptier buckets. If buckets are
                // equally empty, use one with the best score.
                if (lhs_size == rhs_size) {
                  return cmp(lhs, rhs);
                }

                return lhs_size < rhs_size;
              });
  }

  std::vector<uint32_t> new_hashes;

  // We can optionally specify the number of hashes we'd like to be
  // random for a new document. This is to encourage an even distribution
  // among buckets.
  if (num_random_hashes > num_hashes) {
    throw std::invalid_argument(
        "num_random_hashes cannot be greater than num hashes.");
  }

  uint32_t num_informed_hashes = num_hashes - num_random_hashes;
  for (uint32_t i = 0; i < num_informed_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes.push_back(hash);
  }

  uint32_t num_buckets = mach_index.numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, num_buckets - 1);
  std::mt19937 rand(global_random::nextSeed());

  for (uint32_t i = 0; i < num_random_hashes; i++) {
    new_hashes.push_back(int_dist(rand));
  }

  return new_hashes;
}

static void addEntityToIndex(uint32_t num_sampled_buckets,
                             uint32_t num_random_buckets,
                             const bolt::Tensor& outputs,
                             dataset::mach::MachIndex& index,
                             uint32_t new_label) {
  std::vector<TopKActivationsQueue> top_ks;
  for (uint32_t i = 0; i < outputs.batchSize(); i++) {
    top_ks.push_back(
        outputs.getVector(i).findKLargestActivations(num_sampled_buckets));
  }

  auto hashes = topHashesForDoc(std::move(top_ks), index, num_sampled_buckets,
                                num_random_buckets);

  index.insert(new_label, hashes);
}

}  // namespace thirdai::automl::mach
