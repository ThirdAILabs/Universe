#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

namespace thirdai::search {

class Blolt {
 public:
  Blolt(uint64_t estimated_dataset_size, uint8_t num_classifiers,
        uint64_t input_dim, uint64_t seed = std::rand())
      : _num_classes(
            std::max<uint64_t>(10000, std::sqrt(estimated_dataset_size))),
        _input_dim(input_dim),
        _seed(seed),
        _num_classifiers(num_classifiers) {
    for (uint8_t classifier_id = 0; classifier_id < _num_classifiers;
         classifier_id++) {
      _classifiers.push_back(
          createBloltClassifierDenseInput(/* input_dim = */ _input_dim,
                                          /* num_classes = */ _num_classes,
                                          /* last_layer_sparsity = */ 0.01));
    }
    // for (auto& classifier : _classifiers) {
    //   classifier.enableSparseInference(/* remember_mistakes = */ false);
    // }
  }

  // #TODO(josh) : Change num_epochs to just train until not converged
  template <typename BATCH_T>
  void index(
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
      const std::vector<std::vector<uint64_t>>& near_neighbor_ids,
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& entire_dataset,
      uint32_t num_epochs_per_iteration = 20, uint32_t num_iterations = 10,
      float learning_rate = 0.1,
      uint32_t num_alternative_groups_to_consider = 5) {
    if (near_neighbor_ids.size() != train_data->len()) {
      throw std::invalid_argument(
          "The near neighbor vector must be the same length as the input "
          "dataset.");
    }

    _all_groups.clear();
    _total_num_points = entire_dataset->len();
    std::mt19937 rng(_seed);

    // (classifier_id, point_id) -> (group_id) in [0, _num_classes)
    std::vector<std::vector<uint64_t>> assignments = getRandomGroupAssignments(
        entire_dataset->len(), _num_classes, rng, _num_classifiers);

    for (uint32_t iteration = 0; iteration < num_iterations; iteration++) {
      // Train
      for (uint8_t classifier_id = 0; classifier_id < _num_classifiers;
           classifier_id++) {
        auto& classifier = _classifiers.at(classifier_id);
        for (uint32_t i = 0; i < num_epochs_per_iteration; i++) {
          classifier.train(
              /* train_data = */ train_data,
              /* train_labels = */
              neighborsToLabels(train_data, assignments.at(classifier_id),
                                near_neighbor_ids,
                                /* num_neighbors_per_batch = */ 10),
              /* loss_fn = */ bolt::BinaryCrossEntropyLoss(),
              /* learning_rate = */ learning_rate,
              /* epochs = */ 1,
              /* rehash = */ 6400, /* rebuild = */ 128000,
              /* metric_names = */ {}, /* verbose = */ true);
          classifier.predict(
              /* test_data = */ train_data,
              /* labels = */
              neighborsToLabels(train_data, assignments.at(classifier_id),
                                near_neighbor_ids,
                                /* num_neighbors_per_batch = */ 10),
              /* output_active_neurons = */ nullptr,
              /* output_activations = */ nullptr,
              /* metric_names = */ {"categorical_accuracy"},
              /* verbose = */ true,
              /* batch_limit = */ 5);
        }
      }

      // Print accuracy
      if (iteration == num_iterations - 1) {
        break;
      }
      // printAccuracy(train_data, near_neighbor_ids);

      // Reassign groups
      assignments = assignGroupsUsingCurrentClassifiers(
          num_alternative_groups_to_consider, entire_dataset);
    }
  }

  std::vector<std::vector<uint64_t>> query(const bolt::BoltBatch& batch,
                                           uint32_t top_k,
                                           int16_t threshold_to_return = -1) {
    if (threshold_to_return == -1) {
      threshold_to_return = _num_classifiers;
    }
    std::vector<bolt::BoltBatch> all_predictions;
    for (auto& classifier : _classifiers) {
      all_predictions.push_back(classifier.predict(batch));
    }
    std::vector<std::vector<uint64_t>> result(batch.getBatchSize());
#pragma omp parallel for
    for (uint64_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
      // TODO(josh): We can speed this up if neccesary
      std::vector<std::pair<float, uint64_t>> group_activation_pairs;
      for (uint8_t classifier_id = 0; classifier_id < _num_classifiers;
           classifier_id++) {
        const bolt::BoltVector currentBoltVector =
            all_predictions.at(classifier_id)[vec_id];
        for (uint64_t i = 0; i < currentBoltVector.len; i++) {
          if (currentBoltVector.isDense()) {
            group_activation_pairs.emplace_back(
                currentBoltVector.activations[i],
                classifier_id * _num_classes + i);
          } else {
            group_activation_pairs.emplace_back(
                currentBoltVector.activations[i],
                classifier_id * _num_classes +
                    currentBoltVector.active_neurons[i]);
          }
        }
      }
      std::sort(group_activation_pairs.begin(), group_activation_pairs.end(),
                std::greater<>());
      std::vector<uint64_t> sorted_group_ids;
      sorted_group_ids.reserve(group_activation_pairs.size());
      for (auto& group_activation_pair : group_activation_pairs) {
        sorted_group_ids.push_back(group_activation_pair.second);
      }
      result.at(vec_id) =
          groupTestingInference(sorted_group_ids, top_k, threshold_to_return,
                                _total_num_points, _all_groups);
    }
    return result;
  }

  uint64_t getInputDim() const { return _input_dim; }

 protected:
  // This needs to be protected since it's a top level serialization target
  // called by a child class, but DO NOT call it unless you are creating a
  // temporary object to serialize into.
  Blolt(){};

 private:
  uint64_t _num_classes, _input_dim, _total_num_points, _seed;
  uint8_t _num_classifiers;
  std::vector<bolt::FullyConnectedNetwork> _classifiers;
  std::vector<std::vector<uint64_t>> _all_groups;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_classes, _input_dim, _total_num_points, _seed,
            _num_classifiers, _classifiers, _all_groups);
  }

  std::vector<std::vector<uint64_t>> assignGroupsUsingCurrentClassifiers(
      uint32_t num_alternative_groups_to_consider,
      const dataset::BoltDatasetPtr& entire_dataset) {
    std::vector<std::vector<uint64_t>> assignments(_num_classifiers);
    for (uint8_t classifier_id = 0; classifier_id < _num_classifiers;
         classifier_id++) {
      auto& classifier = _classifiers.at(classifier_id);
      std::vector<uint64_t> new_group_assignments;
      std::vector<uint64_t> new_group_sizes(_num_classes, 0);
      for (const auto& batch : *entire_dataset) {
        auto prediction = classifier.predict(batch);
        updateGroupAssigmentsForBatch(new_group_assignments, new_group_sizes,
                                      prediction,
                                      num_alternative_groups_to_consider);
      }
      printGroupSizeProperties(new_group_sizes);
      assignments.at(classifier_id) = new_group_assignments;
    }
    buildGroups(assignments);
    return assignments;
  }

  void buildGroups(const std::vector<std::vector<uint64_t>>& assignments) {
    _all_groups.clear();
    for (const std::vector<uint64_t>& classifier_assignments : assignments) {
      std::vector<std::vector<uint64_t>> classifier_groups(_num_classes);
      for (uint64_t i = 0; i < classifier_assignments.size(); i++) {
        classifier_groups.at(classifier_assignments[i]).push_back(i);
      }
      _all_groups.insert(_all_groups.end(), classifier_groups.begin(),
                         classifier_groups.end());
    }
  }

  void printAccuracy(
      const dataset::BoltDatasetPtr& train_data,
      const std::vector<std::vector<uint64_t>>& near_neighbor_ids) {
    uint32_t index = 0;
    float recall = 0;
    for (const auto& batch : *train_data) {
      auto query_results = query(batch, /* top_k = */ 100);
      for (const auto& result : query_results) {
        recall += getRecall(result, near_neighbor_ids.at(index));
        index++;
      }
      std::cout << recall / index << " " << index << std::endl;
    }
    recall /= index;
  }

  static float getRecall(const std::vector<uint64_t>& result,
                         const std::vector<uint64_t>& gt) {
    float total = 0;
    for (uint32_t i = 0; i < 1; i++) {
      for (uint32_t j = 0; j < result.size(); j++) {
        if (gt.at(i) == result.at(j)) {
          total += 1;
          break;
        }
      }
    }
    return total / 10;
  }

  static bolt::FullyConnectedNetwork createBloltClassifierDenseInput(
      uint64_t input_dim, uint64_t num_classes, float last_layer_sparsity = 0.1,
      uint64_t hidden_layer_dim = 1024, float hidden_layer_sparsity = 1) {
    bolt::SequentialConfigList layers;
    layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
        hidden_layer_dim, hidden_layer_sparsity,
        thirdai::bolt::ActivationFunction::ReLU));
    layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
        num_classes, last_layer_sparsity,
        thirdai::bolt::ActivationFunction::Sigmoid));
    return bolt::FullyConnectedNetwork(layers, input_dim);
  }

  static std::vector<std::vector<uint64_t>> getRandomGroupAssignments(
      uint64_t num_items_in_dataset, uint64_t num_groups, std::mt19937 gen,
      uint8_t num_classifiers) {
    std::vector<std::vector<uint64_t>> all_assignments;
    for (uint8_t classifier_id = 0; classifier_id < num_classifiers;
         classifier_id++) {
      std::uniform_int_distribution<uint64_t> distr(0, num_groups - 1);
      std::vector<uint64_t> single_classifier_assignments(num_items_in_dataset);
      std::vector<uint64_t> group_sizes(num_groups, 0);
      for (uint64_t i = 0; i < num_items_in_dataset; i++) {
        single_classifier_assignments[i] = distr(gen);
        group_sizes[single_classifier_assignments[i]]++;
      }
      printGroupSizeProperties(group_sizes);
      all_assignments.push_back(single_classifier_assignments);
    }
    return all_assignments;
  }

  static void updateGroupAssigmentsForBatch(
      std::vector<uint64_t>& assignments_so_far,
      std::vector<uint64_t>& sizes_so_far, const bolt::BoltBatch& prediction,
      uint8_t num_groups_to_consider) {
    for (uint64_t vec_id = 0; vec_id < prediction.getBatchSize(); vec_id++) {
      bolt::BoltVector prediction_vec = prediction[vec_id];
      std::vector<std::pair<float, uint32_t>> sorted_groups;
      for (uint64_t i = 0; i < prediction_vec.len; i++) {
        if (prediction_vec.isDense()) {
          sorted_groups.emplace_back(prediction_vec.activations[i], i);
        } else {
          sorted_groups.emplace_back(prediction_vec.activations[i],
                                     prediction_vec.active_neurons[i]);
        }
      }
      std::sort(sorted_groups.begin(), sorted_groups.end(), std::greater<>());

      uint64_t chosen_group = 0;
      uint32_t chosen_group_size = UINT32_MAX;
      for (uint64_t i = 0;
           i < std::min<uint64_t>(num_groups_to_consider, sorted_groups.size());
           i++) {
        uint32_t current_group = sorted_groups.at(i).second;
        uint32_t current_group_size = sizes_so_far.at(current_group);
        if (current_group_size < chosen_group_size) {
          chosen_group = current_group;
          chosen_group_size = current_group_size;
        }
      }
      assert(chosen_group_size != UINT32_MAX);

      sizes_so_far.at(chosen_group)++;
      assignments_so_far.push_back(chosen_group);
    }
  }

  static std::vector<uint64_t> groupTestingInference(
      const std::vector<uint64_t>& sorted_group_ids, uint32_t top_k,
      uint8_t replication_threshold, uint64_t total_num_points,
      const std::vector<std::vector<uint64_t>>& groups) {
    std::vector<uint64_t> result;
    std::vector<uint8_t> point_counts(total_num_points, 0);
    for (uint64_t group_id : sorted_group_ids) {
      // std::cout << group_id << " ";
      for (uint64_t point_id : groups[group_id]) {
        point_counts[point_id]++;
        if (point_counts[point_id] == replication_threshold) {
          result.push_back(point_id);
          if (result.size() == top_k) {
            // std::cout << std::endl;
            return result;
          }
        }
      }
    }
    // std::cout << std::endl;
    return result;
  }

  static dataset::BoltDatasetPtr neighborsToLabels(
      const dataset::BoltDatasetPtr& train,
      const std::vector<uint64_t>& group_assignments,
      const std::vector<std::vector<uint64_t>>& near_neighbor_ids,
      const uint32_t num_neighbors_per_batch) {
    std::vector<bolt::BoltBatch> batches;
    uint64_t current_index = 0;
    for (uint32_t batch = 0; batch < train->numBatches(); batch++) {
      uint32_t batch_size = train->at(batch).getBatchSize();
      // TODO(josh): Clean up this dim hack
      bolt::BoltBatch label_batch(
          /* dim = */ num_neighbors_per_batch,
          /* batch_size = */ batch_size,
          /* is_dense = */ false);
      for (uint64_t i = 0; i < batch_size; i++) {
        for (uint64_t d = 0; d < num_neighbors_per_batch; d++) {
          // TODO(josh): Check for repeats
          // std::cout << current_index << " " <<
          // group_assignments[current_index] << " " <<
          // near_neighbor_ids.at(current_index).at(d) << " " <<
          // group_assignments.at(near_neighbor_ids.at(current_index).at(d)) <<
          // std::endl;
          label_batch[i].active_neurons[d] =
              group_assignments.at(near_neighbor_ids.at(current_index).at(d));
          label_batch[i].activations[d] = 1.0;
        }
        current_index++;
      }
      batches.push_back(std::move(label_batch));
    }

    return std::make_shared<dataset::BoltDataset>(std::move(batches),
                                                  current_index);
  }

  static void printGroupSizeProperties(
      const std::vector<uint64_t>& group_sizes) {
    uint64_t sum =
        std::accumulate(std::begin(group_sizes), std::end(group_sizes), 0UL);
    double mean = sum / static_cast<float>(group_sizes.size());
    uint64_t min = *std::min_element(group_sizes.begin(), group_sizes.end());
    uint64_t max = *std::max_element(group_sizes.begin(), group_sizes.end());
    double accum = 0.0;
    std::for_each(std::begin(group_sizes), std::end(group_sizes),
                  [&](const double d) { accum += (d - mean) * (d - mean); });
    double stdev = sqrt(accum / (group_sizes.size() - 1));
    std::cout << "STDDEV: " << stdev << ", MIN: " << min << ", MAX: " << max
              << std::endl;
  }
};
}  // namespace thirdai::search