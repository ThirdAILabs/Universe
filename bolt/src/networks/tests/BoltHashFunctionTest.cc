// #include "BoltNetworkTestUtils.h"
// #include <bolt/src/layers/LayerConfig.h>
// #include <bolt/src/layers/LayerUtils.h>
// #include <bolt/src/networks/FullyConnectedNetwork.h>
// #include <gtest/gtest.h>
// #include <algorithm>
// #include <optional>
// #include <random>
// #include <vector>

// namespace thirdai::bolt::tests {

// static constexpr uint32_t n_classes = 100;

// static void testSimpleDatasetHashFunction(const std::string& hash_function) {
//   // As we train for more epochs, the model should learn better using these hash
//   // functions.
//   FullyConnectedNetwork network(
//       {std::make_shared<FullyConnectedLayerConfig>(
//            /*dim = */ 10000, /*sparsity = */ 0.1,
//            /*act_func = */ ActivationFunction::ReLU,
//            /*sampling_config = */
//            SamplingConfig(/*hashes_per_table = */ 5, /*num_tables = */ 64,
//                           /*range_pow = */ 15, /*reservoir size = */ 4,
//                           /*hash_function = */ hash_function)),
//        std::make_shared<FullyConnectedLayerConfig>(
//            n_classes, ActivationFunction::Softmax)},
//       n_classes);

//   auto [data, labels] =
//       genDataset(/* n_classes= */ n_classes, /* noisy_dataset = */ false);

//   // train the network for two epochs
//   network.train(data, labels, CategoricalCrossEntropyLoss(),
//                 /*learning_rate = */ 0.001, /*epochs = */ 2,
//                 /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
//                 /* verbose= */ false);
//   auto first_test_metrics =
//       network.predict(data, labels, /* output_active_neurons= */ nullptr,
//                       /* output_activations= */ nullptr,
//                       /* use_sparse_inference= */ false,
//                       /* metric_names= */ {"categorical_accuracy"},
//                       /* verbose= */ false);

//   // train the network for 5 epochs
//   network.train(data, labels, CategoricalCrossEntropyLoss(),
//                 /*learning_rate = */ 0.001, /*epochs = */ 5,
//                 /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
//                 /* verbose= */ false);
//   auto second_test_metrics =
//       network.predict(data, labels, /* output_active_neurons= */ nullptr,
//                       /* output_activations= */ nullptr,
//                       /* use_sparse_inference= */ false,
//                       /* metric_names= */ {"categorical_accuracy"},
//                       /* verbose= */ false);

//   // assert that the accuracy improves.
//   ASSERT_GE(second_test_metrics["categorical_accuracy"],
//             first_test_metrics["categorical_accuracy"]);
// }

// // test for DWTA Hash Function
// TEST(BoltHashFunctionTest, TrainSimpleDatasetDWTA) {
//   testSimpleDatasetHashFunction("DWTA");
// }

// // test for SRP Hash Function
// TEST(BoltHashFunctionTest, TrainSimpleDatasetSRP) {
//   testSimpleDatasetHashFunction("SRP");
// }

// // test for FastSRP Hash Function
// TEST(BoltHashFunctionTest, TrainSimpleDatasetFastSRP) {
//   testSimpleDatasetHashFunction("FastSRP");
// }

// }  // namespace thirdai::bolt::tests