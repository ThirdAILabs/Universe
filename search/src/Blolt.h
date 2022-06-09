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
        uint64_t input_dim, uint8_t num_alternative_groups_to_consider = 5,
        uint64_t seed = std::rand())
      : _num_classes(std::sqrt(estimated_dataset_size)),
        _input_dim(input_dim),
        _seed(seed),
        _num_classifiers(num_classifiers),
        _num_alternative_groups_to_consider(
            num_alternative_groups_to_consider) {
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
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& rest_of_data,
      uint32_t num_epochs = 10, float learning_rate = 0.01) {
    (void)_num_alternative_groups_to_consider;
    _groups.clear();
    _total_num_points = train_data->len() + rest_of_data->len();
    std::mt19937 rng(_seed);

    for (auto& classifier : _classifiers) {
      std::vector<uint64_t> current_assignments =
          getRandomGroupAssignments(_total_num_points, _num_classes, rng);
      for (uint32_t epoch = 0; epoch < num_epochs; epoch++) {
        classifier.train(
            /* train_data = */ train_data,
            /* train_labels = */ labelsToBoltbatch(current_assignments),
            /* loss_fn = */ bolt::CategoricalCrossEntropyLoss(),
            /* learning_rate = */ learning_rate, /* epochs = */ 1,
            /* rehash = */ 1, /* rebuild = */ UINT32_MAX,
            /* metric_names = */ {}, /* verbose = */ false);
        // classifier.predict()

        // current_assignments[classifier_id] =
        // self._get_new_group_assignments(
        //     predicted_group_ids=predictions[1],
        //     predicted_activations=predictions[2],
        //     num_groups_to_consider=self.num_groups_to_consider
        // )

        // all_predictions = classifier.predict(dataset, None, 2048)
        // all_assignments = self._get_new_group_assignments(
        //     predicted_group_ids=all_predictions[1],
        //     predicted_activations=all_predictions[2],
        //     num_groups_to_consider=1
        // )
        // group_memberships = [[] for _ in range(self.num_classes)]
        // group_lens = [0 for _ in range(self.num_classes)]
        // for vec_id, group_id in enumerate(all_assignments):
        //     group_memberships[group_id].append(vec_id)
        //     group_lens[group_id] += 1
        // self.groups += group_memberships
      }
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
              currentBoltVector.active_neurons[i]);
        }
      }
      std::sort(group_activation_pairs.begin(), group_activation_pairs.end(),
                std::greater<>());
      std::vector<uint64_t> sorted_group_ids(group_activation_pairs.size());
      for (auto& group_activation_pair : group_activation_pairs) {
        sorted_group_ids.push_back(group_activation_pair.second);
      }
      result.push_back(_group_testing_inference(sorted_group_ids, top_k,
                                                threshold_to_return,
                                                _total_num_points, _groups));
    }
    return result;
  }

  uint64_t getInputDim() const { return _input_dim; }

 private:
  uint64_t _num_classes, _input_dim, _total_num_points, _seed;
  uint8_t _num_classifiers, _num_alternative_groups_to_consider;
  std::vector<bolt::FullyConnectedNetwork> _classifiers;
  std::vector<std::vector<uint64_t>> _groups;

  static bolt::FullyConnectedNetwork createBloltClassifierDenseInput(
      uint64_t input_dim, uint64_t num_classes,
      float last_layer_sparsity = 0.025, uint64_t hidden_layer_dim = 10000,
      float hidden_layer_sparsity = 0.01) {
    bolt::SequentialConfigList layers;
    layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
        hidden_layer_dim, hidden_layer_sparsity,
        thirdai::bolt::ActivationFunction::ReLU));
    layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
        num_classes, last_layer_sparsity,
        thirdai::bolt::ActivationFunction::ReLU));
    return bolt::FullyConnectedNetwork(layers, input_dim);
  }

  static std::vector<uint64_t> getRandomGroupAssignments(
      uint64_t num_items_in_dataset, uint64_t num_groups, std::mt19937 gen) {
    std::uniform_int_distribution<uint64_t> distr(0, num_groups);
    std::vector<uint64_t> random_group_assignments(num_items_in_dataset);
    for (uint64_t i = 0; i < num_items_in_dataset; i++) {
      random_group_assignments[i] = distr(gen);
    }
    return random_group_assignments;
  }

  static std::vector<uint64_t> getNewGroupAssignment(
      bolt::BoltBatch prediction, uint8_t num_groups_to_consider,
      uint64_t num_classes) {
    std::vector<uint64_t> new_group_sizes(num_classes, 0);
    std::vector<uint64_t> new_group_assignments;

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
      float chosen_group_activation = 0;
      for (uint64_t i = 0;
           i < std::min<uint64_t>(num_groups_to_consider, sorted_groups.size());
           i++) {
        if (sorted_groups[i].first > chosen_group_activation) {
          chosen_group = sorted_groups[i].second;
          chosen_group_activation = sorted_groups[i].first;
        }
      }

      new_group_sizes[chosen_group]++;
      new_group_assignments.push_back(chosen_group);
    }

    return new_group_assignments;
  }

  static std::vector<uint64_t> _group_testing_inference(
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

  static dataset::BoltDatasetPtr labelsToBoltbatch(
      const std::vector<uint64_t>& labels) {
    bolt::BoltBatch result(/* dim = */ 1, /* batch_size = */ labels.size(),
                           /* is_dense = */ false);
    for (uint64_t i = 0; i < labels.size(); i++) {
      result[i].active_neurons[0] = labels[i];
      result[i].activations[0] = 1.0;
    }
    std::vector<bolt::BoltBatch> batches;
    batches.push_back(std::move(result));
    return std::make_shared<dataset::BoltDataset>(std::move(batches),
                                                  labels.size());
  }
};
}  // namespace thirdai::search