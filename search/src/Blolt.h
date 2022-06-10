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

namespace thirdai::search {

class Blolt {
 public:
  Blolt(uint64_t estimated_dataset_size, uint8_t num_classifiers,
        uint64_t input_dim, uint64_t seed = std::rand())
      : _num_classes(std::sqrt(estimated_dataset_size)),
        _input_dim(input_dim),
        _seed(seed),
        _num_classifiers(num_classifiers) {
    for (uint8_t classifier_id = 0; classifier_id < _num_classifiers;
         classifier_id++) {
      _classifiers.push_back(
          createBloltClassifierDenseInput(/* input_dim = */ _input_dim,
                                          /* num_classes = */ _num_classes));
    }
    for (auto& classifier : _classifiers) {
      classifier.enableSparseInference(/* remember_mistakes = */ false);
    }
  }

  // #TODO(josh) : Change num_epochs to just train until not converged
  template <typename BATCH_T>
  void index(
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& entire_dataset,
      uint32_t num_epochs = 5, float learning_rate = 0.01,
      uint32_t num_alternative_groups_to_consider = 5) {
    _all_groups.clear();
    _total_num_points = entire_dataset->len();
    std::mt19937 rng(_seed);

    for (auto& classifier : _classifiers) {
      std::vector<uint64_t> current_assignments =
          getRandomGroupAssignments(_total_num_points, _num_classes, rng);
      for (uint32_t epoch = 0; epoch < num_epochs; epoch++) {
        std::vector<std::string> metrics = {"categorical_accuracy"};
        classifier.train(
            /* train_data = */ train_data,
            /* train_labels = */
            labelsToBoltDataset(current_assignments, train_data),
            /* loss_fn = */ bolt::CategoricalCrossEntropyLoss(),
            /* learning_rate = */ learning_rate, /* epochs = */ 10,
            /* rehash = */ UINT32_MAX, /* rebuild = */ UINT32_MAX,
            /* metric_names = */ metrics, /* verbose = */ true);
        if (epoch == num_epochs - 1) {
          break;
        }
        std::vector<uint64_t> new_group_assignments;
        std::vector<uint64_t> new_group_sizes(_num_classes, 0);
        for (uint32_t batch_id = 0; batch_id < train_data->numBatches();
             batch_id++) {
          auto prediction = classifier.predict(train_data->at(batch_id));
          updateGroupAssigmentsForBatch(new_group_assignments, new_group_sizes,
                                        prediction,
                                        num_alternative_groups_to_consider);
        }
        printGroupSizeProperties(new_group_sizes);
        current_assignments = new_group_assignments;
      }

      std::vector<uint64_t> new_group_assignments;
      std::vector<uint64_t> new_group_sizes(_num_classes, 0);
      for (uint32_t batch_id = 0; batch_id < entire_dataset->numBatches();
           batch_id++) {
        auto prediction = classifier.predict(entire_dataset->at(batch_id));
        updateGroupAssigmentsForBatch(new_group_assignments, new_group_sizes,
                                      prediction, 1);
      }
      printGroupSizeProperties(new_group_sizes);

      std::vector<std::vector<uint64_t>> groups(_num_classes);
      for (uint64_t i = 0; i < new_group_assignments.size(); i++) {
        groups.at(new_group_assignments[i]).push_back(i);
      }
      _all_groups.insert(_all_groups.end(), groups.begin(), groups.end());
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
    std::vector<std::vector<uint64_t>> result;
    for (uint64_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
      // TODO(josh): We can speed this up if neccesary
      std::vector<std::pair<float, uint64_t>> group_activation_pairs;
      for (uint8_t classifier_id = 0; classifier_id < _num_classifiers;
           classifier_id++) {
        const bolt::BoltVector currentBoltVector =
            all_predictions.at(classifier_id)[vec_id];
        for (uint64_t i = 0; i < currentBoltVector.len; i++) {
          group_activation_pairs.emplace_back(
              currentBoltVector.activations[i],
              classifier_id * _num_classes +
                  currentBoltVector.active_neurons[i]);
        }
      }
      std::sort(group_activation_pairs.begin(), group_activation_pairs.end(),
                std::greater<>());
      std::vector<uint64_t> sorted_group_ids;
      sorted_group_ids.reserve(group_activation_pairs.size());
      for (auto& group_activation_pair : group_activation_pairs) {
        sorted_group_ids.push_back(group_activation_pair.second);
      }
      result.push_back(groupTestingInference(sorted_group_ids, top_k,
                                             threshold_to_return,
                                             _total_num_points, _all_groups));
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

  static bolt::FullyConnectedNetwork createBloltClassifierDenseInput(
      uint64_t input_dim, uint64_t num_classes,
      float last_layer_sparsity = 0.025, uint64_t hidden_layer_dim = 1000,
      float hidden_layer_sparsity = 0.1) {
    bolt::SequentialConfigList layers;
    layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
        hidden_layer_dim, hidden_layer_sparsity,
        thirdai::bolt::ActivationFunction::ReLU));
    layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
        num_classes, last_layer_sparsity,
        thirdai::bolt::ActivationFunction::Softmax));
    return bolt::FullyConnectedNetwork(layers, input_dim);
  }

  static std::vector<uint64_t> getRandomGroupAssignments(
      uint64_t num_items_in_dataset, uint64_t num_groups, std::mt19937 gen) {
    std::uniform_int_distribution<uint64_t> distr(0, num_groups - 1);
    std::vector<uint64_t> random_group_assignments(num_items_in_dataset);
    std::vector<uint64_t> group_sizes(num_groups, 0);
    for (uint64_t i = 0; i < num_items_in_dataset; i++) {
      random_group_assignments[i] = distr(gen);
      group_sizes[random_group_assignments[i]]++;
    }
    printGroupSizeProperties(group_sizes);
    return random_group_assignments;
  }

  static void updateGroupAssigmentsForBatch(
      std::vector<uint64_t>& assignments_so_far,
      std::vector<uint64_t>& sizes_so_far, const bolt::BoltBatch& prediction,
      uint8_t num_groups_to_consider) {
    for (uint64_t vec_id = 0; vec_id < prediction.getBatchSize(); vec_id++) {
      bolt::BoltVector prediction_vec = prediction[vec_id];
      float* activations = prediction_vec.activations;
      uint32_t* group_ids = prediction_vec.active_neurons;

      std::vector<std::pair<float, uint32_t>> sorted_groups;
      for (uint64_t i = 0; i < prediction_vec.len; i++) {
        sorted_groups.emplace_back(activations[i], group_ids[i]);
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
      for (uint64_t point_id : groups[group_id]) {
        point_counts[point_id]++;
        if (point_counts[point_id] == replication_threshold) {
          result.push_back(point_id);
          if (result.size() == top_k) {
            return result;
          }
        }
      }
    }
    return result;
  }

  static dataset::BoltDatasetPtr labelsToBoltDataset(
      const std::vector<uint64_t>& labels,
      const dataset::BoltDatasetPtr& train_data) {
    std::vector<bolt::BoltBatch> batches;
    uint64_t current_label_index = 0;
    for (uint32_t batch = 0; batch < train_data->numBatches(); batch++) {
      bolt::BoltBatch label_batch(/* dim = */ 1,
                                  /* batch_size = */ labels.size(),
                                  /* is_dense = */ false);
      for (uint64_t i = 0; i < train_data->at(batch).getBatchSize(); i++) {
        label_batch[i].active_neurons[0] = labels[current_label_index];
        label_batch[i].activations[0] = 1.0;
        current_label_index++;
      }
      batches.push_back(std::move(label_batch));
    }

    return std::make_shared<dataset::BoltDataset>(std::move(batches),
                                                  labels.size());
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